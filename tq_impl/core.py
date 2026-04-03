"""
tq_impl/core.py  —  v2 (bit-packed, dual-mode 3b/4b)
=====================================================

Implements Algorithm 1 (TurboQuant_mse) and Algorithm 2 (TurboQuant_prod)
from Zandieh et al. "TurboQuant: Online Vector Quantization for KV Cache
Compression with Near-Optimal Distortion Rate", ICLR 2026.

Key changes from v1:
  - PackedKeys dataclass with bit-packed uint8 storage
  - Support for both 3-bit (2b MSE + 1b QJL) and 4-bit (3b MSE + 1b QJL)
  - MSE-only dequantize path for standard attention (lower noise)
  - Fused score path for decode (no decompression)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .codebook import get_codebook, get_boundaries
from .bitpack import pack_2bit, unpack_2bit, pack_3bit, unpack_3bit, pack_1bit, unpack_1bit


# ---------------------------------------------------------------------------
# Packed data container
# ---------------------------------------------------------------------------

@dataclass
class PackedKeys:
    """
    Bit-packed compressed keys from TurboQuantProd.

    Storage (for D=128):
      3-bit mode (2b MSE + 1b QJL):  32 + 16 + 4 = 52 bytes/position  (4.9x vs fp16)
      4-bit mode (3b MSE + 1b QJL):  64 + 16 + 4 = 84 bytes/position  (3.0x vs fp16)
    """
    packed_idx:     torch.Tensor   # uint8  [..., D // pack_factor]
    packed_qjl:     torch.Tensor   # uint8  [..., D // 8]
    residual_norm:  torch.Tensor   # fp16   [...]
    key_norm:       torch.Tensor   # fp16   [...]
    head_dim:       int
    bits_mse:       int            # 2 or 3
    bits_total:     float          # 3.0 or 4.0


def concat_packed_seq(a: PackedKeys, b: PackedKeys) -> PackedKeys:
    """Concatenate two PackedKeys along the sequence dimension (dim=-2 for 4D)."""
    return PackedKeys(
        packed_idx=torch.cat([a.packed_idx, b.packed_idx], dim=-2),
        packed_qjl=torch.cat([a.packed_qjl, b.packed_qjl], dim=-2),
        residual_norm=torch.cat([a.residual_norm, b.residual_norm], dim=-1),
        key_norm=torch.cat([a.key_norm, b.key_norm], dim=-1),
        head_dim=a.head_dim,
        bits_mse=a.bits_mse,
        bits_total=a.bits_total,
    )


def reorder_packed(c: PackedKeys, beam_idx: torch.Tensor) -> PackedKeys:
    """Reorder along batch dimension (dim 0) for beam search."""
    return PackedKeys(
        packed_idx=c.packed_idx.index_select(0, beam_idx),
        packed_qjl=c.packed_qjl.index_select(0, beam_idx),
        residual_norm=c.residual_norm.index_select(0, beam_idx),
        key_norm=c.key_norm.index_select(0, beam_idx),
        head_dim=c.head_dim,
        bits_mse=c.bits_mse,
        bits_total=c.bits_total,
    )


def slice_packed(c: PackedKeys, b: int, h: int) -> PackedKeys:
    """Extract [T, ...] slice for batch b, head h from [B, H, T, ...] packed cache."""
    return PackedKeys(
        packed_idx=c.packed_idx[b, h],
        packed_qjl=c.packed_qjl[b, h],
        residual_norm=c.residual_norm[b, h],
        key_norm=c.key_norm[b, h],
        head_dim=c.head_dim,
        bits_mse=c.bits_mse,
        bits_total=c.bits_total,
    )


# ---------------------------------------------------------------------------
# TurboQuant_mse  (Algorithm 1) — internal helper
# ---------------------------------------------------------------------------

class TurboQuantMSE:
    """
    MSE-optimal scalar quantiser per coordinate (Algorithm 1).

    The random rotation Pi decorrelates coordinates so that independent
    scalar quantisation is near-optimal.
    """

    def __init__(
        self,
        bits:     int,
        head_dim: int,
        device:   str = "cuda",
        seed:     Optional[int] = None,
        dtype:    torch.dtype = torch.float16,
    ) -> None:
        self.bits     = bits
        self.head_dim = head_dim
        self.n_levels = 2 ** bits
        self.device   = device
        self.dtype    = dtype

        # Haar random orthogonal rotation via QR
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        raw = torch.randn(head_dim, head_dim, generator=gen)
        Pi, _ = torch.linalg.qr(raw)
        self.Pi = Pi.to(device=device, dtype=dtype)

        # Lloyd-Max codebook
        self.centroids  = get_codebook(bits, head_dim).to(device=device, dtype=dtype)
        self.boundaries = get_boundaries(bits, head_dim).to(device=device, dtype=dtype)

    def quantize_raw(self, x_unit: torch.Tensor) -> torch.Tensor:
        """
        Quantize unit-norm vectors, return raw indices (int16).

        x_unit: [..., D] unit-norm vectors
        Returns: [..., D] int16 indices in [0, n_levels)
        """
        *lead, d = x_unit.shape
        x_f = x_unit.reshape(-1, d).to(self.dtype)
        y = x_f @ self.Pi.T
        idx = torch.bucketize(y, self.boundaries)
        return idx.to(torch.int16).reshape(*lead, d)

    def dequantize_from_idx(
        self, idx: torch.Tensor, key_norm: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Reconstruct vectors from raw indices.

        idx:      [..., D] int16
        key_norm: [...]    fp16 (optional, applies scaling)
        Returns:  [..., D] reconstructed vectors
        """
        *lead, d = idx.shape
        idx_f = idx.reshape(-1, d).to(torch.int64)
        y_hat = self.centroids[idx_f]
        x_hat = (y_hat @ self.Pi).to(self.dtype)

        if key_norm is not None:
            norms = key_norm.reshape(-1).to(self.dtype)
            x_hat = x_hat * norms.unsqueeze(-1)

        return x_hat.reshape(*lead, d)


# ---------------------------------------------------------------------------
# TurboQuant_prod  (Algorithm 2)
# ---------------------------------------------------------------------------

class TurboQuantProd:
    """
    Inner-product-optimal vector quantiser (Algorithm 2).

    Parameters
    ----------
    bits     : total effective bits per coordinate
               3.0 → 2-bit MSE + 1-bit QJL  (4.9x key compression at D=128)
               4.0 → 3-bit MSE + 1-bit QJL  (3.0x key compression at D=128)
    head_dim : vector dimension
    device   : 'cuda' or 'cpu'
    seed     : RNG seed
    dtype    : compute dtype
    """

    def __init__(
        self,
        bits:     float = 4.0,
        head_dim: int   = 128,
        device:   str   = "cuda",
        seed:     Optional[int] = None,
        dtype:    torch.dtype   = torch.float16,
    ) -> None:
        self.bits     = bits
        self.head_dim = head_dim
        self.device   = device
        self.dtype    = dtype
        self.bits_mse = max(1, int(math.floor(bits)) - 1)

        self.mse = TurboQuantMSE(
            bits=self.bits_mse, head_dim=head_dim,
            device=device, seed=seed, dtype=dtype,
        )

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed((seed or 0) + 1337)
        self.S = torch.randn(
            head_dim, head_dim, generator=gen
        ).to(device=device, dtype=dtype)

        self._qjl_const = math.sqrt(math.pi / 2) / head_dim

    # ------------------------------------------------------------------
    # Quantize → PackedKeys
    # ------------------------------------------------------------------

    def quantize(self, x: torch.Tensor) -> PackedKeys:
        """
        Compress vectors to bit-packed representation.

        x: [..., head_dim]
        Returns: PackedKeys with actual bit-packed uint8 storage
        """
        *leading, d = x.shape
        assert d == self.head_dim

        x_f = x.reshape(-1, d).to(self.dtype)
        key_norms = x_f.norm(dim=-1)
        x_hat = x_f / (key_norms.unsqueeze(-1) + 1e-8)

        # Stage 1: MSE quantisation
        idx_raw = self.mse.quantize_raw(x_hat)
        x_mse   = self.mse.dequantize_from_idx(idx_raw)

        # Stage 2: QJL on residual
        residual  = x_hat - x_mse
        res_norms = residual.norm(dim=-1)
        Sr  = residual @ self.S.T
        qjl = torch.sign(Sr).to(torch.int8)
        qjl = qjl.masked_fill(qjl == 0, 1)

        # Bit-pack
        N = idx_raw.shape[0]
        if self.bits_mse == 2:
            packed_idx = pack_2bit(idx_raw.reshape(N, d))
        elif self.bits_mse == 3:
            packed_idx = pack_3bit(idx_raw.reshape(N, d))
        else:
            packed_idx = idx_raw.reshape(N, d).to(torch.uint8)

        packed_qjl = pack_1bit(qjl.reshape(N, d))

        # Reshape to match leading dims
        pack_d_idx = packed_idx.shape[-1]
        pack_d_qjl = packed_qjl.shape[-1]

        return PackedKeys(
            packed_idx=packed_idx.reshape(*leading, pack_d_idx),
            packed_qjl=packed_qjl.reshape(*leading, pack_d_qjl),
            residual_norm=res_norms.to(torch.float16).reshape(*leading),
            key_norm=key_norms.to(torch.float16).reshape(*leading),
            head_dim=d,
            bits_mse=self.bits_mse,
            bits_total=self.bits,
        )

    # ------------------------------------------------------------------
    # Dequantize — MSE-only (for standard attention)
    # ------------------------------------------------------------------

    def dequantize_mse(self, pk: PackedKeys) -> torch.Tensor:
        """
        Reconstruct using MSE stage only (no QJL noise).
        Best quality for standard Q @ K^T attention path.
        """
        idx = self._unpack_idx(pk)
        return self.mse.dequantize_from_idx(idx, key_norm=pk.key_norm)

    # ------------------------------------------------------------------
    # Dequantize — full Prod (for debugging/comparison)
    # ------------------------------------------------------------------

    def dequantize_full(self, pk: PackedKeys) -> torch.Tensor:
        """
        Full TurboQuant_prod reconstruction with QJL correction.
        Unbiased inner products but noisier reconstruction.
        """
        idx = self._unpack_idx(pk)
        qjl = self._unpack_qjl(pk)

        *lead, d = idx.shape
        N = idx.reshape(-1, d).shape[0]

        x_mse = self.mse.dequantize_from_idx(idx.reshape(-1, d))
        qjl_f = qjl.reshape(N, d)
        res_n = pk.residual_norm.reshape(N).to(self.dtype)
        key_n = pk.key_norm.reshape(N).to(self.dtype)

        correction = (qjl_f @ self.S) * (self._qjl_const * res_n.unsqueeze(-1))
        x_hat  = x_mse + correction
        x_full = x_hat * key_n.unsqueeze(-1)
        return x_full.reshape(*lead, d)

    # ------------------------------------------------------------------
    # Fused score — no decompression
    # ------------------------------------------------------------------

    def score_fused(
        self,
        query: torch.Tensor,    # [D] or [B, D]
        pk:    PackedKeys,
    ) -> torch.Tensor:
        """
        Compute attention logits directly on packed data.

        score_i = ||k_i|| * ||q|| * [<Pq_hat, c_{idx_i}> + const * gamma_i * <Sq_hat, b_i>]
        """
        d = self.head_dim
        q_2d   = query.unsqueeze(0) if query.dim() == 1 else query
        q_norm = q_2d.norm(dim=-1, keepdim=True)
        q_unit = (q_2d / (q_norm + 1e-8)).to(self.dtype)

        Pq = q_unit @ self.mse.Pi.T
        Sq = q_unit @ self.S.T

        idx = self._unpack_idx(pk)
        qjl = self._unpack_qjl(pk)

        *leading, d2 = idx.shape
        assert d2 == d
        N = math.prod(leading) if leading else 1

        idx_f  = idx.reshape(N, d).to(torch.int64)
        qjl_f  = qjl.reshape(N, d)
        res_n  = pk.residual_norm.reshape(N).to(self.dtype)
        key_n  = pk.key_norm.reshape(N).to(self.dtype)

        c_lut      = self.mse.centroids[idx_f]
        mse_scores = torch.einsum("bd,nd->bn", Pq, c_lut)

        qjl_scores = torch.einsum("bd,nd->bn", Sq, qjl_f)
        qjl_corr   = self._qjl_const * res_n.unsqueeze(0) * qjl_scores

        scores = (mse_scores + qjl_corr) * key_n.unsqueeze(0) * q_norm

        if query.dim() == 1:
            return scores.reshape(*leading)
        return scores.reshape(q_2d.shape[0], *leading)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unpack_idx(self, pk: PackedKeys) -> torch.Tensor:
        if pk.bits_mse == 2:
            return unpack_2bit(pk.packed_idx, pk.head_dim)
        elif pk.bits_mse == 3:
            return unpack_3bit(pk.packed_idx, pk.head_dim)
        return pk.packed_idx.to(torch.int16)

    def _unpack_qjl(self, pk: PackedKeys) -> torch.Tensor:
        return unpack_1bit(pk.packed_qjl, pk.head_dim)
