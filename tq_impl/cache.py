"""
tq_impl/cache.py  —  v3 (bit-packed, prefill-aware)
====================================================

Two-phase KV cache using bit-packed TurboQuant keys.

Phase 1 (Prefill):  Raw FP16 keys → exact attention, zero quality loss
Phase 2 (Decode):   Bit-packed keys → 4.9x (3b) or 3.0x (4b) key compression

Compatible with HuggingFace transformers >= 4.50 (DynamicCache API).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from .core import TurboQuantProd, PackedKeys, concat_packed_seq, reorder_packed, slice_packed
from .bitpack import packed_bytes_per_position, fp16_bytes_per_position


class TurboQuantCache:
    """
    Drop-in HuggingFace KV cache with TurboQuant key compression.

    Usage:
        cache = TurboQuantCache(bits=4)  # 4-bit mode (3b MSE + 1b QJL)
        output = model.generate(input_ids, past_key_values=cache, ...)

    Parameters
    ----------
    bits  : 3.0 (2b MSE + 1b QJL, 4.9x compression) or
            4.0 (3b MSE + 1b QJL, 3.0x compression, better quality)
    dtype : working dtype (default float16)
    seed  : RNG seed for reproducibility
    """

    def __init__(
        self,
        bits:  float = 4.0,
        dtype: torch.dtype = torch.float16,
        seed:  Optional[int] = 42,
    ) -> None:
        self.bits  = bits
        self.dtype = dtype
        self.seed  = seed

        self._quantisers: Dict[int, TurboQuantProd] = {}

        # Phase 1: raw FP16 (prefill)
        self._raw_keys: Dict[int, torch.Tensor] = {}

        # Phase 2: bit-packed (decode)
        self._packed_keys: Dict[int, PackedKeys] = {}

        # Values always FP16
        self._values: Dict[int, torch.Tensor] = {}

        self._compressed: Dict[int, bool] = {}
        self._seen_tokens: int = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_q(self, layer_idx: int, head_dim: int, device: str) -> TurboQuantProd:
        if layer_idx not in self._quantisers:
            self._quantisers[layer_idx] = TurboQuantProd(
                bits=self.bits, head_dim=head_dim,
                device=device, seed=(self.seed or 0) + layer_idx * 1000,
                dtype=self.dtype,
            )
        return self._quantisers[layer_idx]

    def _compress_layer(self, layer_idx: int) -> None:
        """Compress prefill raw keys → bit-packed format."""
        if self._compressed.get(layer_idx):
            return
        raw = self._raw_keys.pop(layer_idx, None)
        if raw is None:
            return
        q = self._quantisers[layer_idx]
        self._packed_keys[layer_idx] = q.quantize(raw)
        self._compressed[layer_idx] = True

    # ------------------------------------------------------------------
    # HuggingFace cache API
    # ------------------------------------------------------------------

    def update(
        self,
        key_states:   torch.Tensor,
        value_states: torch.Tensor,
        layer_idx:    int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        B, H, T_new, D = key_states.shape
        q = self._get_q(layer_idx, D, str(key_states.device))

        # Values
        v = value_states.to(self.dtype)
        if layer_idx not in self._values:
            self._values[layer_idx] = v
        else:
            self._values[layer_idx] = torch.cat([self._values[layer_idx], v], dim=2)

        is_prefill = T_new > 1

        if is_prefill:
            # PREFILL: store raw, return exact
            k = key_states.to(self.dtype)
            if layer_idx not in self._raw_keys:
                self._raw_keys[layer_idx] = k
            else:
                self._raw_keys[layer_idx] = torch.cat(
                    [self._raw_keys[layer_idx], k], dim=2
                )
            return self._raw_keys[layer_idx], self._values[layer_idx]
        else:
            # DECODE: compress
            if not self._compressed.get(layer_idx):
                self._compress_layer(layer_idx)

            new_pk = q.quantize(key_states)

            if layer_idx in self._packed_keys:
                self._packed_keys[layer_idx] = concat_packed_seq(
                    self._packed_keys[layer_idx], new_pk
                )
            else:
                self._packed_keys[layer_idx] = new_pk

            # Dequantize MSE-only for standard attention
            full_keys = q.dequantize_mse(self._packed_keys[layer_idx])
            return full_keys, self._values[layer_idx]

    def update_compressed(
        self,
        key_states:   torch.Tensor,
        value_states: torch.Tensor,
        layer_idx:    int,
    ) -> torch.Tensor:
        """
        Update cache WITHOUT dequantizing keys — for fused scoring mode.

        Returns only values tensor. Keys remain bit-packed in self._packed_keys.
        This avoids allocating a full FP16 key tensor, giving real VRAM savings.
        """
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        B, H, T_new, D = key_states.shape
        q = self._get_q(layer_idx, D, str(key_states.device))

        # Values
        v = value_states.to(self.dtype)
        if layer_idx not in self._values:
            self._values[layer_idx] = v
        else:
            self._values[layer_idx] = torch.cat([self._values[layer_idx], v], dim=2)

        is_prefill = T_new > 1

        if is_prefill:
            # PREFILL: store raw (needed for exact attention during prefill)
            k = key_states.to(self.dtype)
            if layer_idx not in self._raw_keys:
                self._raw_keys[layer_idx] = k
            else:
                self._raw_keys[layer_idx] = torch.cat(
                    [self._raw_keys[layer_idx], k], dim=2
                )
        else:
            # DECODE: compress only, NO dequantize
            if not self._compressed.get(layer_idx):
                self._compress_layer(layer_idx)

            new_pk = q.quantize(key_states)

            if layer_idx in self._packed_keys:
                self._packed_keys[layer_idx] = concat_packed_seq(
                    self._packed_keys[layer_idx], new_pk
                )
            else:
                self._packed_keys[layer_idx] = new_pk

        return self._values[layer_idx]

    # ------------------------------------------------------------------
    # Fused score (bypasses decompression)
    # ------------------------------------------------------------------

    def fused_scores(
        self,
        query_states: torch.Tensor,   # [B, H_q, 1, D]
        layer_idx:    int,
    ) -> torch.Tensor:
        """Attention logits [B, H_q, 1, T] via fused scoring on packed data."""
        if layer_idx not in self._packed_keys:
            raise ValueError(f"Layer {layer_idx} not compressed")

        B, H_q, _, D = query_states.shape
        q = self._quantisers[layer_idx]
        pk = self._packed_keys[layer_idx]
        H_kv = pk.packed_idx.shape[1]          # number of KV heads (e.g. 4)
        gqa_ratio = max(1, H_q // H_kv)        # GQA group size (e.g. 7)

        q_flat = query_states.reshape(B * H_q, D).to(self.dtype)

        scores_list = []
        for bh in range(B * H_q):
            b_idx  = bh // H_q
            h_q    = bh % H_q
            h_kv   = h_q // gqa_ratio           # map Q head → KV head
            q_bh   = q_flat[bh : bh + 1]
            pk_bh  = slice_packed(pk, b=b_idx, h=h_kv)
            s = q.score_fused(q_bh, pk_bh)
            scores_list.append(s)

        scores = torch.cat(scores_list, dim=0)
        T_k = pk.packed_idx.shape[2]
        return scores.reshape(B, H_q, 1, T_k)

    # ------------------------------------------------------------------
    # HF compatibility
    # ------------------------------------------------------------------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx in self._packed_keys:
            return self._packed_keys[layer_idx].packed_idx.shape[2]
        if layer_idx in self._raw_keys:
            return self._raw_keys[layer_idx].shape[2]
        return 0

    def get_max_length(self) -> Optional[int]:
        return None

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int = 0) -> Tuple[int, int]:
        return self.get_seq_length(layer_idx) + cache_position.shape[0], 0

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    @property
    def key_cache(self) -> List[torch.Tensor]:
        out = []
        for i in sorted(set(list(self._raw_keys) + list(self._packed_keys))):
            if i in self._raw_keys:
                out.append(self._raw_keys[i])
            elif i in self._packed_keys:
                out.append(self._quantisers[i].dequantize_mse(self._packed_keys[i]))
        return out

    @property
    def value_cache(self) -> List[torch.Tensor]:
        return [self._values[i] for i in sorted(self._values)]

    def __len__(self) -> int:
        return max(len(self._raw_keys), len(self._packed_keys), len(self._values))

    def __iter__(self):
        for i in sorted(self._values):
            if i in self._raw_keys:
                yield self._raw_keys[i], self._values[i]
            elif i in self._packed_keys:
                yield self._quantisers[i].dequantize_mse(self._packed_keys[i]), self._values[i]

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        for li in list(self._raw_keys):
            self._raw_keys[li] = self._raw_keys[li].index_select(0, beam_idx)
        for li in list(self._packed_keys):
            self._packed_keys[li] = reorder_packed(self._packed_keys[li], beam_idx)
        for li in list(self._values):
            self._values[li] = self._values[li].index_select(0, beam_idx)

    def crop(self, max_length: int) -> None:
        for li in list(self._raw_keys):
            if self._raw_keys[li].shape[2] > max_length:
                self._raw_keys[li] = self._raw_keys[li][:, :, :max_length, :]
        for li in list(self._packed_keys):
            pk = self._packed_keys[li]
            if pk.packed_idx.shape[2] > max_length:
                self._packed_keys[li] = PackedKeys(
                    packed_idx=pk.packed_idx[:, :, :max_length, :],
                    packed_qjl=pk.packed_qjl[:, :, :max_length, :],
                    residual_norm=pk.residual_norm[:, :, :max_length],
                    key_norm=pk.key_norm[:, :, :max_length],
                    head_dim=pk.head_dim, bits_mse=pk.bits_mse, bits_total=pk.bits_total,
                )
        for li in list(self._values):
            if self._values[li].shape[2] > max_length:
                self._values[li] = self._values[li][:, :, :max_length, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        for li in list(self._raw_keys):
            self._raw_keys[li] = self._raw_keys[li].repeat_interleave(repeats, dim=0)
        for li in list(self._packed_keys):
            pk = self._packed_keys[li]
            self._packed_keys[li] = PackedKeys(
                packed_idx=pk.packed_idx.repeat_interleave(repeats, dim=0),
                packed_qjl=pk.packed_qjl.repeat_interleave(repeats, dim=0),
                residual_norm=pk.residual_norm.repeat_interleave(repeats, dim=0),
                key_norm=pk.key_norm.repeat_interleave(repeats, dim=0),
                head_dim=pk.head_dim, bits_mse=pk.bits_mse, bits_total=pk.bits_total,
            )
        for li in list(self._values):
            self._values[li] = self._values[li].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for li in list(self._raw_keys):
            self._raw_keys[li] = self._raw_keys[li][indices]
        for li in list(self._packed_keys):
            self._packed_keys[li] = reorder_packed(self._packed_keys[li], indices)
        for li in list(self._values):
            self._values[li] = self._values[li][indices]

    def reset(self) -> None:
        self._raw_keys.clear()
        self._packed_keys.clear()
        self._values.clear()
        self._compressed.clear()
        self._quantisers.clear()
        self._seen_tokens = 0

    # ------------------------------------------------------------------
    # Memory diagnostics
    # ------------------------------------------------------------------

    def memory_footprint(self) -> dict:
        """Report actual byte-level memory usage."""
        raw_bytes = sum(t.nelement() * 2 for t in self._raw_keys.values())

        packed_bytes = 0
        for pk in self._packed_keys.values():
            packed_bytes += pk.packed_idx.nelement()     # uint8
            packed_bytes += pk.packed_qjl.nelement()     # uint8
            packed_bytes += pk.residual_norm.nelement() * 2  # fp16
            packed_bytes += pk.key_norm.nelement() * 2       # fp16

        value_bytes = sum(t.nelement() * 2 for t in self._values.values())

        # Equivalent FP16 keys
        total_positions = 0
        head_dim = 128
        for pk in self._packed_keys.values():
            total_positions += pk.key_norm.nelement()
            head_dim = pk.head_dim
        for t in self._raw_keys.values():
            total_positions += t.shape[0] * t.shape[1] * t.shape[2]
            head_dim = t.shape[3]

        fp16_key_bytes = total_positions * head_dim * 2

        return {
            "raw_key_bytes": raw_bytes,
            "packed_key_bytes": packed_bytes,
            "value_bytes": value_bytes,
            "fp16_key_equivalent": fp16_key_bytes,
            "key_compression_ratio": fp16_key_bytes / max(1, packed_bytes + raw_bytes),
            "total_cache_bytes": raw_bytes + packed_bytes + value_bytes,
            "fp16_total_equivalent": fp16_key_bytes + value_bytes,
        }
