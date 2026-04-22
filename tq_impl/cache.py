"""
tq_impl/cache.py  —  v18 (Elite V3 MASTER)
=========================================
Finalized Dual-Space architecture with Full Device Parity.
Supports Heterogeneous Gemma-4 architectures (D=512 fallback).
"""
from __future__ import annotations
import math
import torch
from typing import Optional, Dict, List, Tuple, Union, Any

from .triton_polar import is_triton_available, triton_polar_encode, triton_polar_decode
from .polar_quant import PolarAngleQuantizer
from .value_quant import ValueQuantizer
from .bitpack import pack_1bit, unpack_1bit

class TurboQuantCache:
    is_compileable = False
    is_initialized = True

    def __init__(
        self, bits: Union[float, List[float], Dict[int, float]] = 4.0,
        bits_key: Optional[float] = None, bits_value: Optional[float] = None,
        outliers: bool = True, num_outlier_pairs: int = 16,
        dtype: Optional[torch.dtype] = None, use_fp8: bool = False, seed: Optional[int] = 42,
        max_seq_len: int = 16384 * 8, chunk_size: int = 2048,
    ) -> None:
        self.bits_config = bits; self.bits_key = bits_key; self.bits_value = bits_value
        self.outliers = outliers; 
        self.num_outlier_pairs = num_outlier_pairs; self.dtype = dtype
        self.use_fp8 = use_fp8; self.seed = seed
        self.max_seq_len = max_seq_len; self.chunk_size = chunk_size
        
        v_bits = int(bits_value if bits_value is not None else 8.0)
        self._value_quantizer = ValueQuantizer(bits=v_bits, use_fp8=use_fp8)
        
        self._qjl_projections = {}; self._angle_quantizers = {}; self._permutations = {}
        self._compressed = {}; self._cur_len = {}; self._allocated_len = {}
        self._final_radii_buf = {}; self._packed_angles_buf = {}
        self._angle_offsets = {}; self._total_ang_bytes = {}
        self._packed_qjl_buf = {}; self._qjl_gammas_buf = {}
        self._values_buf = {}; self._value_states_buf = {}
        self._v_rec_cache = {}; self._outlier_indices = {}; 
        self._outlier_vals_buf = {}; self._outlier_idx_buf = {}
        self._raw_keys = {}; self._raw_values = {}
        self._seen_tokens = 0
        self.compress_start = 0
        self._triton_scratches: Dict[torch.device, torch.Tensor] = {}

    def _get_scratch(self, size, device):
        # 🚀 Fix: Dynamic Lean Workspace (v22)
        # Only allocate what is strictly necessary for the current chunk
        if device not in self._triton_scratches or self._triton_scratches[device].shape[0] < size:
            self._triton_scratches[device] = torch.empty(size, device=device, dtype=torch.float32)
        return self._triton_scratches[device][:size]

    def _to_dev(self, tensor, device):
        if tensor is None: return None
        if tensor.device == device: return tensor
        return tensor.to(device)

    def _get_resources(self, i: int, D: int, device: torch.device):
        if i not in self._qjl_projections:
            st = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
            torch.manual_seed((self.seed or 0) + i)
            self._permutations[i] = torch.randperm(D, device=device)
            proj = torch.randn(D, D, device=device, dtype=self.dtype)
            q_orth, _ = torch.linalg.qr(proj.float())
            self._qjl_projections[i] = q_orth.to(device).to(self.dtype)
            self._angle_quantizers[i] = PolarAngleQuantizer(d=D, bits=int(self.bits_config))
            if st is not None: torch.cuda.set_rng_state(st, device)
        return self._angle_quantizers[i], self._to_dev(self._qjl_projections[i], device)

    def _allocate_buffers(self, i, B, H, D, device, initial_len=None):
        needs_realloc = False
        if i in self._packed_angles_buf:
            existing_H = self._packed_angles_buf[i].shape[1]
            existing_D = self._packed_qjl_buf[i].shape[3] * 8
            if existing_H != H or existing_D != D:
                print(f"[TurboQuant Cache] Layer {i} Shift: H={existing_H}->{H}, D={existing_D}->{D}", flush=True)
                needs_realloc = True
        if i not in self._packed_angles_buf or needs_realloc:
            pq, _ = self._get_resources(i, D, device)
            L = int(math.log2(D)); bits = int(self.bits_config); alloc_len = 512
            self._allocated_len[i] = alloc_len; self._cur_len[i] = 0
            self._final_radii_buf[i] = torch.zeros((B, H, alloc_len, 1), device=device, dtype=self.dtype)
            total_ppp = 0; offsets = []
            for lv in range(L):
                lvl_d = D >> (lv + 1); ppp = max(1, (lvl_d * bits) // 8)
                offsets.append(total_ppp); total_ppp += ppp
            self._angle_offsets[i] = torch.tensor(offsets, device=device, dtype=torch.int32)
            self._packed_angles_buf[i] = torch.zeros((B, H, alloc_len, total_ppp), device=device, dtype=torch.uint8)
            self._packed_qjl_buf[i] = torch.zeros((B, H, alloc_len, D // 8), device=device, dtype=torch.uint8)
            self._qjl_gammas_buf[i] = torch.zeros((B, H, alloc_len, 1), device=device, dtype=self.dtype)
            if self.outliers: 
                self._outlier_vals_buf[i] = torch.zeros((B, H, alloc_len, self.num_outlier_pairs * 2), device=device, dtype=self.dtype)
                self._outlier_idx_buf[i] = torch.zeros((B, H, alloc_len, self.num_outlier_pairs), dtype=torch.int16, device=device)
            if self._value_quantizer.bits == 8:
                self._values_buf[i] = torch.zeros((B, H, alloc_len, D), device=device, dtype=torch.int8)
                self._value_states_buf[i] = torch.ones((B, H, alloc_len, 1), device=device, dtype=self.dtype)
            else:
                self._values_buf[i] = torch.zeros((B, H, alloc_len, D), device=device, dtype=self.dtype)

    def _ensure_capacity(self, i, needed):
        if needed <= self._allocated_len.get(i, 0): return
        old_len = self._allocated_len[i]; new_len = min(self.max_seq_len, ((needed + self.chunk_size - 1) // self.chunk_size) * self.chunk_size)
        if new_len <= old_len: return
        def pad(x, nl):
            s = list(x.shape); s[2] = nl - x.shape[2]; return torch.cat([x, torch.zeros(s, device=x.device, dtype=x.dtype)], dim=2)
        self._final_radii_buf[i] = pad(self._final_radii_buf[i], new_len)
        self._packed_angles_buf[i] = pad(self._packed_angles_buf[i], new_len)
        self._packed_qjl_buf[i] = pad(self._packed_qjl_buf[i], new_len)
        self._qjl_gammas_buf[i] = pad(self._qjl_gammas_buf[i], new_len)
        self._values_buf[i] = pad(self._values_buf[i], new_len)
        if i in self._value_states_buf:
            x = self._value_states_buf[i]; s = list(x.shape); s[2] = new_len - x.shape[2]
            self._value_states_buf[i] = torch.cat([x, torch.ones(s, device=x.device, dtype=x.dtype)], dim=2)
        if i in self._outlier_vals_buf: self._outlier_vals_buf[i] = pad(self._outlier_vals_buf[i], new_len)
        self._allocated_len[i] = new_len

    def _extract_outliers(self, k, i):
        if not self.outliers: return k, None, None
        B, H, T, D = k.shape; k_p = k.view(B, H, T, D // 2, 2)
        if i not in self._outlier_indices:
            heavy_idx = torch.topk(torch.linalg.vector_norm(k_p, dim=-1).mean(dim=(0, 2)), self.num_outlier_pairs, dim=1).indices
            forced = torch.arange(4, device=heavy_idx.device).expand(H, 4)
            idx = torch.cat([forced, heavy_idx], dim=1)
            self._outlier_indices[i] = idx[:, :self.num_outlier_pairs]
        idx = self._to_dev(self._outlier_indices[i], k.device)
        if H != idx.shape[0]: idx = idx.repeat_interleave(H // idx.shape[0], dim=0)
        id_ex = idx.view(1, H, 1, self.num_outlier_pairs, 1).expand(B, H, T, self.num_outlier_pairs, 2)
        vals = torch.gather(k_p, 3, id_ex).view(B, H, T, -1)
        start = self._cur_len.get(i, 0); self._outlier_vals_buf[i][:, :, start:start+T, :] = vals
        k_q = k_p.clone(); k_q.scatter_(3, id_ex, 0.0)
        return k_q.view(B, H, T, D), idx, vals

    def update_compressed(self, k, v, i):
        B, H, T, D = k.shape; device = k.device
        if D > 256:
            if i not in self._raw_keys: self._raw_keys[i] = []; self._raw_values[i] = []
            self._raw_keys[i].append(k.to(self.dtype)); self._raw_values[i].append(v.to(self.dtype))
            self._seen_tokens += T; self._cur_len[i] = self._cur_len.get(i, 0) + T; return k
        self._allocate_buffers(i, B, H, D, device)
        self._ensure_capacity(i, self._cur_len[i] + T)
        pq, proj = self._get_resources(i, D, device)
        perm = self._to_dev(self._permutations[i], device); k_perm = k[..., perm].contiguous()
        start = self._cur_len[i]; total = start + T
        kz, _, _ = self._extract_outliers(k_perm, i)
        
        # 🚀 Fix: Revert to safe 16384 stride to prevent Illegal Access
        scratch = self._get_scratch(B * H * T * 16384, device)
        rn, pn = triton_polar_encode(kz, pq.get_all_boundaries(device=device), D, bits=pq.bits, scratch=scratch)
        self._final_radii_buf[i][:, :, start:total, :] = rn
        offs = self._angle_offsets[i]
        for lv, b in enumerate(pn): self._packed_angles_buf[i][:, :, start:total, offs[lv]:offs[lv]+b.shape[-1]] = b
        k_rs = triton_polar_decode(rn, pn, pq.get_all_centroids(device=device), D, bits=pq.bits)
        qjl, g = self._compute_qjl(kz, k_rs, proj)
        self._packed_qjl_buf[i][:, :, start:total, :] = qjl; self._qjl_gammas_buf[i][:, :, start:total, :] = g
        vn, vst = self._value_quantizer.quantize(v); self._values_buf[i][:, :, start:total, :] = vn
        if vst is not None: self._value_states_buf[i][:, :, start:total, :] = vst
        self._cur_len[i] = total
        
        # 🚀 Fix: Prefill Memory Stripping
        # If we are in prefill (T > 1), return the high-fidelity input to save 3GB of reconstruction VRAM
        if T > 1: return k, v
        return self._get_v_rec(i, total, device)

    def _get_v_rec(self, i, total, device=None):
        if i not in self._values_buf and i in self._raw_values:
            return self._to_dev(torch.cat(self._raw_values[i], dim=2), device)
        v_rec = self._value_quantizer.dequantize(self._values_buf[i][:, :, :total, :], self._value_states_buf[i][:, :, :total, :] if i in self._value_states_buf else None, self.dtype)
        if device: v_rec = self._to_dev(v_rec, device)
        self._v_rec_cache[i] = v_rec; return v_rec

    def fused_scores(self, q, i):
        dev = q.device; T = self._cur_len[i]; D = q.shape[-1]; pq, proj = self._get_resources(i, D, dev)
        from .triton_attention import triton_fused_polar_attention_decode
        perm = self._to_dev(self._permutations[i], dev); q_p = q[..., perm].contiguous()
        q_qjl = torch.matmul(q_p, proj).contiguous()
        rf = self._to_dev(self._final_radii_buf[i][:, :, :T, :], dev)
        pa = self._to_dev(self._packed_angles_buf[i][:, :, :T, :], dev)
        off = self._to_dev(self._angle_offsets[i], dev); ct = pq.get_all_centroids(device=dev)
        pqjl = self._to_dev(self._packed_qjl_buf[i][:, :, :T, :], dev)
        g = self._to_dev(self._qjl_gammas_buf[i][:, :, :T, :], dev)
        oi = self._to_dev(self._outlier_indices[i], dev).to(torch.int32)
        ov = self._to_dev(self._outlier_vals_buf[i][:, :, :T, :], dev)
        return triton_fused_polar_attention_decode(q_p, q_qjl, rf, pa, off, ct, oi, ov, pqjl, g, D, pq.bits)

    def _compute_qjl(self, k, k_rs, proj):
        u = torch.matmul(k - k_rs, proj.to(device=k.device, dtype=k.dtype))
        s = torch.sign(u); s = torch.where(s==0, torch.ones_like(s), s)
        return pack_1bit(s.to(torch.int8)), torch.abs(u).mean(dim=-1, keepdim=True)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        i = layer_idx
        B, H, T_new, D = key_states.shape; device = key_states.device
        # 🚀 Optimization: Lean Outliers for Gemma-4
        self.num_outlier_pairs = 8
        if self.dtype is None: self.dtype = key_states.dtype
        if D > 256:
            self.update_compressed(key_states, value_states, i)
            return self._to_dev(torch.cat(self._raw_keys[i], dim=2), device), self._to_dev(torch.cat(self._raw_values[i], dim=2), device)
        if not self._compressed.get(i):
            if self._seen_tokens < self.compress_start:
                self._raw_keys[i] = torch.cat([self._raw_keys.get(i, torch.empty((B, H, 0, D), device=device, dtype=self.dtype)), key_states], dim=2)
                self._raw_values[i] = torch.cat([self._raw_values.get(i, torch.empty((B, H, 0, value_states.shape[-1]), device=device, dtype=self.dtype)), value_states], dim=2)
                if i == 0: self._seen_tokens += T_new
                return self._raw_keys[i], self._raw_values[i]
            else: self._compress_layer(i, key_states, value_states)
        else: self.update_compressed(key_states, value_states, i)
        if i == 0: self._seen_tokens += T_new
        T = self._cur_len[i]; return self._reconstruct_keys(i, T, device), self._get_v_rec(i, T, device)

    def _compress_layer(self, i, k_new, v_new):
        raw_k = torch.cat([self._raw_keys.get(i, torch.empty((k_new.shape[0], k_new.shape[1], 0, k_new.shape[-1]), device=k_new.device, dtype=self.dtype)), k_new], dim=2)
        raw_v = torch.cat([self._raw_values.get(i, torch.empty((v_new.shape[0], v_new.shape[1], 0, v_new.shape[-1]), device=v_new.device, dtype=self.dtype)), v_new], dim=2)
        self.update_compressed(raw_k, raw_v, i); self._compressed[i] = True; self._raw_keys.pop(i, None); self._raw_values.pop(i, None)

    def _reconstruct_keys(self, i, T=None, device=None):
        if i not in self._final_radii_buf:
            if i in self._raw_keys: return self._to_dev(torch.cat(self._raw_keys[i], dim=2), device)
            return None
        if T is None: T = self._cur_len[i]
        B, H, _, _ = self._final_radii_buf[i].shape; D = self._values_buf[i].shape[-1]; L = int(math.log2(D))
        dev = device if device else self._final_radii_buf[i].device
        pq, proj = self._get_resources(i, D, dev)
        rf = self._to_dev(self._final_radii_buf[i][:, :, :T, 0], dev); pa_flat = self._to_dev(self._packed_angles_buf[i][:, :, :T, :], dev)
        D_idx = torch.arange(D, device=dev).view(1, 1, 1, D); radii = rf.unsqueeze(-1).expand(B, H, T, D).clone()
        offsets = self._angle_offsets[i].cpu().tolist(); ct = pq.get_all_centroids(device=dev)
        for lv in range(L-1, -1, -1):
            is_right = (D_idx >> lv) & 1; ang_idx = (D_idx >> (lv + 1))
            byte_off = offsets[lv] + (ang_idx * pq.bits) // 8; bits_shift = (ang_idx * pq.bits) % 8
            bytes_val = torch.gather(pa_flat, 3, byte_off.expand(B, H, T, D))
            q_idx = (bytes_val >> bits_shift) & (0x0F if pq.bits == 4 else 0x07)
            phi = ct[lv][q_idx.long()]; radii *= torch.where(is_right == 1, torch.sin(phi), torch.cos(phi))
        idx = self._to_dev(self._outlier_indices[i], dev); id_ex = idx.view(1, H, 1, self.num_outlier_pairs, 1).expand(B, H, T, self.num_outlier_pairs, 2)
        ov = self._to_dev(self._outlier_vals_buf[i][:, :, :T, :], dev).view(B, H, T, self.num_outlier_pairs, 2)
        k_p = radii.view(B, H, T, D//2, 2); k_p.scatter_(3, id_ex, ov)
        k_rs = k_p.view(B, H, T, D); p_qjl = self._to_dev(self._packed_qjl_buf[i][:, :, :T, :], dev); g = self._to_dev(self._qjl_gammas_buf[i][:, :, :T, :], dev)
        qs = unpack_1bit(p_qjl, D).to(self.dtype); corr = (qs @ proj.T) * g
        k_perm = k_rs + corr; i_perm = torch.argsort(self._to_dev(self._permutations[i], dev)); return k_perm[..., i_perm]

    def get_seq_length(self, layer_idx=0): return self._cur_len.get(layer_idx, 0)
    def get_max_length(self): return self.max_seq_len
    def get_mask_sizes(self, q_len, layer_idx=0): return self.get_seq_length(layer_idx) + (q_len.shape[0] if torch.is_tensor(q_len) else q_len), 0
    def __len__(self): return len(self._cur_len)
    
    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens
        
    def memory_footprint(self) -> int:
        total = 0
        for buf_dict in [self._final_radii_buf, self._packed_angles_buf, self._packed_qjl_buf, 
                         self._qjl_gammas_buf, self._values_buf, self._value_states_buf, 
                         self._outlier_vals_buf, self._outlier_idx_buf]:
            for v in buf_dict.values():
                if v is not None and hasattr(v, 'element_size'):
                    total += v.nelement() * v.element_size()
        return total