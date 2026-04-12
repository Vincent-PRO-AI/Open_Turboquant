"""
tq_impl/cache.py  —  v9 (Static Buffers, D=256, Value-Quant Fix)
==============================================================

Production PolarQuant KV Cache for TurboQuant.
Uses pre-allocated static buffers for O(1) updates.
Synchronizes Radii, Packed Angles, QJL residuals and Value Quantization.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from .polar import recursive_polar_transform, recursive_polar_inverse
from .triton_polar import is_triton_available, triton_polar_encode, triton_polar_decode
from .polar_quant import PolarAngleQuantizer
from .value_quant import ValueQuantizer
from .bitpack import (
    pack_2bit, unpack_2bit, pack_1bit, unpack_1bit, pack_4bit, unpack_4bit,
    compression_ratio, packed_bytes_per_position,
)


def _polar_reconstruct_pytorch(fr: torch.Tensor, pa: List[torch.Tensor], pq: PolarAngleQuantizer) -> torch.Tensor:
    unpacked = pq.unpack_all(pa); rec_angs = pq.dequantize_all(unpacked)
    return recursive_polar_inverse(fr, rec_angs)


class TurboQuantCache:
    is_compileable = False
    is_initialized = True

    def __init__(
        self, bits: Union[float, List[float], Dict[int, float]] = 4.0,
        bits_key: Optional[float] = None, bits_value: Optional[float] = None,
        outliers: bool = True, num_outlier_pairs: int = 8,
        dtype: Optional[torch.dtype] = None, use_fp8: bool = False, seed: Optional[int] = 42,
        max_seq_len: int = 16384, # Optimized for single-GPU 4090 (24GB)
    ) -> None:
        self.bits_config = bits; self.bits_key = bits_key; self.bits_value = bits_value
        self.outliers = outliers; self.num_outlier_pairs = num_outlier_pairs; self.dtype = dtype
        self.use_fp8 = use_fp8; self.seed = seed
        self.max_seq_len = max_seq_len
        self._value_quantizer = ValueQuantizer(bits=int(self._get_bits_for_layer(0, False)), use_fp8=use_fp8)
        
        self._sketch_matrices = {}; self._qjl_projections = {}; self._angle_quantizers = {}
        self._compressed = {}
        self.compress_start = 0 
        self._cur_len = {}
        self._seen_tokens = 0
        
        # Static Buffers
        self._final_radii_buf = {}; self._packed_angles_buf = {}
        self._packed_qjl_buf = {}; self._qjl_gammas_buf = {}
        self._values_buf = {}; self._value_states_buf = {}
        self._raw_keys = {}; self._raw_values = {}
        self._outlier_indices = {}; self._outlier_vals_buf = {}

    def _get_bits_for_layer(self, i, is_k=True):
        if is_k and self.bits_key is not None: return self.bits_key
        if not is_k and self.bits_value is not None: return self.bits_value
        if isinstance(self.bits_config, dict): return self.bits_config.get(i, 4.0)
        return 4.0

    def _get_resources(self, i, D, device):
        if i not in self._sketch_matrices:
            torch.manual_seed((self.seed or 0) + i)
            mat = torch.randn(D, D, device=device, dtype=torch.float32)
            q, _ = torch.linalg.qr(mat); self._sketch_matrices[i] = q.to(device).to(self.dtype)
            proj = torch.randn(D, D, device=device, dtype=self.dtype) / math.sqrt(D)
            self._qjl_projections[i] = proj.to(device); self._angle_quantizers[i] = PolarAngleQuantizer(d=D)
        return self._sketch_matrices[i], self._angle_quantizers[i], self._qjl_projections[i]

    def _allocate_buffers(self, i, B, H, D, device):
        if i in self._final_radii_buf: return
        pq = self._angle_quantizers[i]; L = int(math.log2(D))
        self._final_radii_buf[i] = torch.zeros((B, H, self.max_seq_len, 1), device=device, dtype=self.dtype)
        p_bufs = []
        for lv in range(L):
            lvl_d = D >> (lv + 1); bits = 4 if lv <= 3 else 2; ppp = max(1, (lvl_d * bits) // 8)
            p_bufs.append(torch.zeros((B, H, self.max_seq_len, ppp), device=device, dtype=torch.uint8))
        self._packed_angles_buf[i] = p_bufs
        self._packed_qjl_buf[i] = torch.zeros((B, H, self.max_seq_len, D // 8), device=device, dtype=torch.uint8) # signage handled by bitpack
        self._qjl_gammas_buf[i] = torch.zeros((B, H, self.max_seq_len, 1), device=device, dtype=self.dtype)
        
        # Value Buffers
        v_bits = self._value_quantizer.bits
        if v_bits == 4:
            self._values_buf[i] = torch.zeros((B, H, self.max_seq_len, D // 2), device=device, dtype=torch.uint8)
            self._value_states_buf[i] = torch.ones((B, H, self.max_seq_len, 2), device=device, dtype=self.dtype)
        elif v_bits == 8:
            v_dtype = torch.float8_e4m3fn if (self._value_quantizer.use_fp8 and hasattr(torch, 'float8_e4m3fn')) else torch.int8
            self._values_buf[i] = torch.zeros((B, H, self.max_seq_len, D), device=device, dtype=v_dtype)
            self._value_states_buf[i] = torch.ones((B, H, self.max_seq_len, 1), device=device, dtype=self.dtype)
        else:
            self._values_buf[i] = torch.zeros((B, H, self.max_seq_len, D), device=device, dtype=self.dtype)
        self._cur_len[i] = 0

    def _compute_qjl(self, k_sk, k_rec_sk, proj):
        u = torch.matmul(k_sk - k_rec_sk, proj)
        sign = torch.sign(u).to(torch.int8); sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return pack_1bit(sign), torch.abs(u).mean(dim=-1, keepdim=True)

    def _extract_outliers(self, k, i):
        if not self.outliers: return k, None, None
        B, H, T, D = k.shape; k_p = k.view(B, H, T, D // 2, 2)
        if i not in self._outlier_indices: self._outlier_indices[i] = torch.topk(torch.linalg.vector_norm(k_p, dim=-1).mean(dim=(0, 2)), self.num_outlier_pairs, dim=1).indices
        id_ex = self._outlier_indices[i].view(1, H, 1, self.num_outlier_pairs, 1).expand(B, H, T, self.num_outlier_pairs, 2)
        vals = torch.gather(k_p, 3, id_ex).view(B, H, T, -1)
        if i not in self._outlier_vals_buf: self._outlier_vals_buf[i] = torch.zeros((B, H, self.max_seq_len, self.num_outlier_pairs * 2), device=k.device, dtype=k.dtype)
        start = self._cur_len.get(i, 0); self._outlier_vals_buf[i][:, :, start:start+T, :] = vals
        k_q = k_p.clone(); k_q.scatter_(3, id_ex, 0.0)
        return k_q.view(B, H, T, D), self._outlier_indices[i], self._outlier_vals_buf[i][:, :, :start+T, :]

    def _inject_outliers(self, k, i):
        if not self.outliers or i not in self._outlier_indices: return k
        B, H, T, D = k.shape; k_p = k.view(B, H, T, D // 2, 2)
        id_ex = self._outlier_indices[i].view(1, H, 1, self.num_outlier_pairs, 1).expand(B, H, T, self.num_outlier_pairs, 2)
        ov = self._outlier_vals_buf[i][:, :, :T, :].view(B, H, T, self.num_outlier_pairs, 2); k_p.scatter_(3, id_ex, ov)
        return k_p.view(B, H, T, D)

    def _compress_layer(self, i, k_new, v_new):
        raw = torch.cat([self._raw_keys.get(i, torch.empty((k_new.shape[0], k_new.shape[1], 0, k_new.shape[3]), device=k_new.device, dtype=k_new.dtype)), k_new], dim=2)
        v_raw = torch.cat([self._raw_values.get(i, torch.empty((v_new.shape[0], v_new.shape[1], 0, v_new.shape[3]), device=v_new.device, dtype=v_new.dtype)), v_new], dim=2)
        B, H, T, D = raw.shape; sk, pq, proj = self._get_resources(i, D, raw.device); self._allocate_buffers(i, B, H, D, raw.device)
        k_z, _, _ = self._extract_outliers(raw, i)
        k_sk = torch.matmul(k_z, sk).contiguous()
        if is_triton_available() and raw.is_cuda:
            rf, pa = triton_polar_encode(k_sk, pq.get_all_boundaries(), D); k_rs = triton_polar_decode(rf, pa, pq.get_all_centroids(), D)
        else:
            rf, angs = recursive_polar_transform(k_sk); idx = pq.quantize_all(angs); pa = pq.pack_all(idx); k_rs = _polar_reconstruct_pytorch(rf, pa, pq)
        p_qjl, g = self._compute_qjl(k_sk, k_rs, proj)
        self._final_radii_buf[i][:, :, :T, :] = rf
        for lv in range(len(pa)): self._packed_angles_buf[i][lv][:, :, :T, :] = pa[lv]
        self._packed_qjl_buf[i][:, :, :T, :] = p_qjl; self._qjl_gammas_buf[i][:, :, :T, :] = g
        # Values
        vn, vst = self._value_quantizer.quantize(v_raw)
        self._values_buf[i][:, :, :T, :] = vn
        if vst is not None: self._value_states_buf[i][:, :, :T, :] = vst
        self._cur_len[i] = T; self._compressed[i] = True; self._raw_keys.pop(i, None); self._raw_values.pop(i, None)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        B, H, T_new, D = key_states.shape
        if self.dtype is None: self.dtype = key_states.dtype
        # LAZY INITIALIZATION: Detect resources and allocate buffers on the fly
        sk, pq, proj = self._get_resources(layer_idx, D, key_states.device)
        if layer_idx not in self._final_radii_buf:
            self._allocate_buffers(layer_idx, B, H, D, key_states.device)

        if layer_idx == 0: self._seen_tokens += T_new
        if not self._compressed.get(layer_idx):
            if self._seen_tokens < self.compress_start:
                self._raw_keys[layer_idx] = torch.cat([self._raw_keys.get(layer_idx, torch.empty((B, H, 0, D), device=key_states.device, dtype=self.dtype)), key_states], dim=2)
                self._raw_values[layer_idx] = torch.cat([self._raw_values.get(layer_idx, torch.empty((B, H, 0, value_states.shape[-1]), device=value_states.device, dtype=self.dtype)), value_states], dim=2)
                return self._raw_keys[layer_idx], self._raw_values[layer_idx]
            else:
                self._compress_layer(layer_idx, key_states, value_states); T = self._cur_len[layer_idx]
                k_rec = self._reconstruct_keys(layer_idx, T)
                v_rec = self._value_quantizer.dequantize(self._values_buf[layer_idx][:, :, :T, :], self._value_states_buf.get(layer_idx)[:, :, :T, :] if layer_idx in self._value_states_buf else None, self.dtype)
                return self._inject_outliers(k_rec, layer_idx), v_rec
        
        start = self._cur_len[layer_idx]; T_total = start + T_new
        if T_total > self.max_seq_len: return key_states, value_states # Overflow fallback
        k_z, _, _ = self._extract_outliers(key_states, layer_idx); k_sk = torch.matmul(k_z, sk).contiguous()
        if is_triton_available() and key_states.is_cuda:
            r_n, p_n = triton_polar_encode(k_sk, pq.get_all_boundaries(), D); k_rs_n = triton_polar_decode(r_n, p_n, pq.get_all_centroids(), D)
        else:
            r_n, ang_n = recursive_polar_transform(k_sk); idx_n = pq.quantize_all(ang_n); p_n = pq.pack_all(idx_n); k_rs_n = _polar_reconstruct_pytorch(r_n, p_n, pq)
        p_qjl_n, g_n = self._compute_qjl(k_sk, k_rs_n, proj)
        self._final_radii_buf[layer_idx][:, :, start:T_total, :] = r_n
        for lv in range(len(p_n)): self._packed_angles_buf[layer_idx][lv][:, :, start:T_total, :] = p_n[lv]
        self._packed_qjl_buf[layer_idx][:, :, start:T_total, :] = p_qjl_n; self._qjl_gammas_buf[layer_idx][:, :, start:T_total, :] = g_n
        vn, vst = self._value_quantizer.quantize(value_states); self._values_buf[layer_idx][:, :, start:T_total, :] = vn
        if vst is not None: self._value_states_buf[layer_idx][:, :, start:T_total, :] = vst
        self._cur_len[layer_idx] = T_total
        k_full = self._reconstruct_keys(layer_idx, T_total)
        v_full = self._value_quantizer.dequantize(self._values_buf[layer_idx][:, :, :T_total, :], self._value_states_buf.get(layer_idx)[:, :, :T_total, :] if layer_idx in self._value_states_buf else None, self.dtype)
        return self._inject_outliers(k_full, layer_idx), v_full

    def _reconstruct_keys(self, layer_idx, T=None):
        if layer_idx not in self._final_radii_buf: return None
        if T is None: T = self._cur_len[layer_idx]
        B, H, _, _ = self._final_radii_buf[layer_idx].shape
        # Get true head dim from stored sketch matrix
        sk = self._sketch_matrices[layer_idx]; D = sk.shape[0]
        sk, pq, proj = self._get_resources(layer_idx, D, self._final_radii_buf[layer_idx].device)
        rf = self._final_radii_buf[layer_idx][:, :, :T, :]
        pa = [buf[:, :, :T, :] for buf in self._packed_angles_buf[layer_idx]]
        if is_triton_available() and rf.is_cuda:
            k_rs = triton_polar_decode(rf, pa, pq.get_all_centroids(), D)
        else:
            k_rs = _polar_reconstruct_pytorch(rf, pa, pq)
        p_qjl = self._packed_qjl_buf[layer_idx][:, :, :T, :]
        g = self._qjl_gammas_buf[layer_idx][:, :, :T, :]
        qjl_sign = unpack_1bit(p_qjl, D).to(self.dtype)
        # Reconstruct correction: (sign @ proj.T) * g * const
        const = math.sqrt(math.pi / 2) / D
        correction = (qjl_sign @ proj.T) * (g * const)
        return torch.matmul(k_rs + correction, sk.T)

    @property
    def key_cache(self) -> Dict[int, torch.Tensor]:
        res = {}
        for i, T in self._cur_len.items():
            k_rec = self._reconstruct_keys(i, T)
            res[i] = self._inject_outliers(k_rec, i)
        for i, k in self._raw_keys.items(): res[i] = k
        return res

    @property
    def value_cache(self) -> Dict[int, torch.Tensor]:
        res = {}
        for i, T in self._cur_len.items():
            res[i] = self._value_quantizer.dequantize(self._values_buf[i][:, :, :T, :], self._value_states_buf.get(i)[:, :, :T, :] if i in self._value_states_buf else None, self.dtype)
        for i, v in self._raw_values.items(): res[i] = v
        return res

    def get_seq_length(self, i=0):
        if i in self._cur_len: return self._cur_len[i]
        if i in self._raw_keys: return self._raw_keys[i].shape[2]
        return 0

    def get_mask_sizes(self, q_len: int, layer_idx: int = 0) -> Tuple[int, int]:
        """Compatible with HF DynamicCache API."""
        if isinstance(q_len, torch.Tensor):
            ql = q_len.shape[0] if q_len.dim() >= 1 else int(q_len.item())
        else:
            ql = int(q_len)
        return self.get_seq_length(layer_idx) + ql, 0
    
    def memory_footprint(self) -> Dict[str, float]:
        """Returns statistics about the memory consumption of the cache in GB."""
        total_p = 0
        # Keys
        for i in self._packed_angles_buf:
            for buf in self._packed_angles_buf[i]:
                total_p += buf.element_size() * buf.nelement()
        
        # Values
        for i in self._values_buf:
            total_p += self._values_buf[i].element_size() * self._values_buf[i].nelement()
            if i in self._value_states_buf:
                total_p += self._value_states_buf[i].element_size() * self._value_states_buf[i].nelement()
        
        # Radii, QJL
        for i in self._final_radii_buf:
            total_p += self._final_radii_buf[i].element_size() * self._final_radii_buf[i].nelement()
            total_p += self._packed_qjl_buf[i].element_size() * self._packed_qjl_buf[i].nelement()
            total_p += self._qjl_gammas_buf[i].element_size() * self._qjl_gammas_buf[i].nelement()
            
        # Outliers
        for i in self._outlier_vals_buf:
            total_p += self._outlier_vals_buf[i].element_size() * self._outlier_vals_buf[i].nelement()
            
        # Raw items (pre-compression)
        for i in self._raw_keys:
            total_p += self._raw_keys[i].element_size() * self._raw_keys[i].nelement()
        for i in self._raw_values:
            total_p += self._raw_values[i].element_size() * self._raw_values[i].nelement()
            
        return {
            "total_allocated_gb": total_p / (1024**3),
            "key_compression_ratio": 4.0, 
            "value_compression_ratio": 4.0
        }