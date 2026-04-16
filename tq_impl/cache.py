"""
tq_impl/cache.py  —  v11 (Elite Accuracy Restored)
=================================================

KV Cache implementation for TurboQuant PolarQuant v2.
Supports dynamic bit-depth per layer/type and high-fidelity residuals.
"""
from __future__ import annotations
import math
import torch
from typing import Optional, Dict, List, Tuple, Union, Any

from .polar import recursive_polar_transform, recursive_polar_inverse
from .triton_polar import is_triton_available, triton_polar_encode, triton_polar_decode
from .polar_quant import PolarAngleQuantizer
from .value_quant import ValueQuantizer
from .bitpack import pack_1bit, unpack_1bit, compression_ratio

class TurboQuantCache:
    is_compileable = False
    is_initialized = True

    def __init__(
        self, bits: Union[float, List[float], Dict[int, float]] = 4.0,
        bits_key: Optional[float] = None, bits_value: Optional[float] = None,
        outliers: bool = True, num_outlier_pairs: int = 8,
        dtype: Optional[torch.dtype] = None, use_fp8: bool = False, seed: Optional[int] = 42,
        max_seq_len: int = 16384 * 8, chunk_size: int = 2048,
    ) -> None:
        self.bits_config = bits; self.bits_key = bits_key; self.bits_value = bits_value
        self.outliers = outliers; self.num_outlier_pairs = num_outlier_pairs; self.dtype = dtype
        self.use_fp8 = use_fp8; self.seed = seed
        self.max_seq_len = max_seq_len; self.chunk_size = chunk_size
        
        v_bits = int(bits_value if bits_value is not None else 8.0)
        self._value_quantizer = ValueQuantizer(bits=v_bits, use_fp8=use_fp8)
        
        self._sketch_matrices = {}; self._qjl_projections = {}; self._angle_quantizers = {}
        self._compressed = {}; self._cur_len = {}; self._allocated_len = {}
        self._final_radii_buf = {}; self._packed_angles_buf = {}
        self._packed_qjl_buf = {}; self._qjl_gammas_buf = {}
        self._values_buf = {}; self._value_states_buf = {}
        self._raw_keys = {}; self._raw_values = {}
        self._outlier_indices = {}; self._outlier_vals_buf = {}
        self._k_rec_cache = {}
        self._seen_tokens = 0
        self.compress_start = 0

    def _get_bits_for_layer(self, i: int, is_k: bool = True) -> int:
        if is_k and self.bits_key is not None: return int(self.bits_key)
        if not is_k and self.bits_value is not None: return int(self.bits_value)
        if isinstance(self.bits_config, dict): return int(self.bits_config.get(i, 4))
        return int(self.bits_config)

    def _get_resources(self, i: int, D: int, device: torch.device):
        if i not in self._sketch_matrices:
            st = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
            torch.manual_seed((self.seed or 0) + i)
            mat = torch.randn(D, D, device=device, dtype=torch.float32)
            q, _ = torch.linalg.qr(mat); self._sketch_matrices[i] = q.to(device).to(self.dtype)
            proj = torch.randn(D, D, device=device, dtype=self.dtype) / math.sqrt(D)
            self._qjl_projections[i] = proj.to(device)
            self._angle_quantizers[i] = PolarAngleQuantizer(d=D, bits=self._get_bits_for_layer(i, True))
            if st is not None: torch.cuda.set_rng_state(st, device)
        return self._sketch_matrices[i], self._angle_quantizers[i], self._qjl_projections[i]

    def _allocate_buffers(self, i, B, H, D, device, initial_len=None):
        if i in self._final_radii_buf: return
        pq = self._angle_quantizers[i]; L = int(math.log2(D))
        bits = self._get_bits_for_layer(i, True)
        alloc_len = min(self.max_seq_len, initial_len if initial_len else self.chunk_size)
        self._allocated_len[i] = alloc_len

        self._final_radii_buf[i] = torch.zeros((B, H, alloc_len, 1), device=device, dtype=self.dtype)
        p_bufs = []
        for lv in range(L):
            lvl_d = D >> (lv + 1); ppp = max(1, (lvl_d * bits) // 8)
            p_bufs.append(torch.zeros((B, H, alloc_len, ppp), device=device, dtype=torch.uint8))
        self._packed_angles_buf[i] = p_bufs
        self._packed_qjl_buf[i] = torch.zeros((B, H, alloc_len, D // 8), device=device, dtype=torch.uint8)
        self._qjl_gammas_buf[i] = torch.zeros((B, H, alloc_len, 1), device=device, dtype=self.dtype)
        
        # Values
        v_bits = self._value_quantizer.bits
        if v_bits == 4:
            self._values_buf[i] = torch.zeros((B, H, alloc_len, D // 2), device=device, dtype=torch.uint8)
            self._value_states_buf[i] = torch.ones((B, H, alloc_len, 2), device=device, dtype=self.dtype)
        elif v_bits == 8:
            # 8-bit still needs a 1-dim scale factor
            self._values_buf[i] = torch.zeros((B, H, alloc_len, D), device=device, dtype=torch.int8)
            self._value_states_buf[i] = torch.ones((B, H, alloc_len, 1), device=device, dtype=self.dtype)
        else:
            self._values_buf[i] = torch.zeros((B, H, alloc_len, D), device=device, dtype=self.dtype)
        
        if self.outliers:
            self._outlier_vals_buf[i] = torch.zeros((B, H, alloc_len, self.num_outlier_pairs * 2), device=device, dtype=self.dtype)
        self._cur_len[i] = 0

    def _ensure_capacity(self, i, needed):
        if needed <= self._allocated_len.get(i, 0): return
        old_len = self._allocated_len[i]
        new_len = min(self.max_seq_len, ((needed + self.chunk_size - 1) // self.chunk_size) * self.chunk_size)
        if new_len <= old_len: return

        print(f"[TurboQuant] Expanding Layer {i} cache: {old_len} -> {new_len}")
        def pad(x, nl):
            s = list(x.shape); s[2] = nl - x.shape[2]
            return torch.cat([x, torch.zeros(s, device=x.device, dtype=x.dtype)], dim=2)

        self._final_radii_buf[i] = pad(self._final_radii_buf[i], new_len)
        for lv in range(len(self._packed_angles_buf[i])):
            self._packed_angles_buf[i][lv] = pad(self._packed_angles_buf[i][lv], new_len)
        self._packed_qjl_buf[i] = pad(self._packed_qjl_buf[i], new_len)
        self._qjl_gammas_buf[i] = pad(self._qjl_gammas_buf[i], new_len)
        self._values_buf[i] = pad(self._values_buf[i], new_len)
        if i in self._value_states_buf:
            x = self._value_states_buf[i]; s = list(x.shape); s[2] = new_len - x.shape[2]
            self._value_states_buf[i] = torch.cat([x, torch.ones(s, device=x.device, dtype=x.dtype)], dim=2)
        if i in self._outlier_vals_buf:
            self._outlier_vals_buf[i] = pad(self._outlier_vals_buf[i], new_len)
        self._allocated_len[i] = new_len

    def _extract_outliers(self, k, i):
        if not self.outliers: return k, None, None
        B, H, T, D = k.shape; k_p = k.view(B, H, T, D // 2, 2)
        if i not in self._outlier_indices: 
            self._outlier_indices[i] = torch.topk(torch.linalg.vector_norm(k_p, dim=-1).mean(dim=(0, 2)), self.num_outlier_pairs, dim=1).indices
        id_ex = self._outlier_indices[i].view(1, H, 1, self.num_outlier_pairs, 1).expand(B, H, T, self.num_outlier_pairs, 2)
        vals = torch.gather(k_p, 3, id_ex).view(B, H, T, -1)
        start = self._cur_len.get(i, 0); self._outlier_vals_buf[i][:, :, start:start+T, :] = vals
        k_q = k_p.clone(); k_q.scatter_(3, id_ex, 0.0)
        return k_q.view(B, H, T, D), self._outlier_indices[i], self._outlier_vals_buf[i][:, :, :start+T, :]

    def _inject_outliers(self, k, i):
        if not self.outliers or i not in self._outlier_indices: return k
        B, H, T, D = k.shape; k_p = k.view(B, H, T, D // 2, 2)
        id_ex = self._outlier_indices[i].view(1, H, 1, self.num_outlier_pairs, 1).expand(B, H, T, self.num_outlier_pairs, 2)
        ov = self._outlier_vals_buf[i][:, :, :T, :].view(B, H, T, self.num_outlier_pairs, 2)
        k_p.scatter_(3, id_ex, ov)
        return k_p.view(B, H, T, D)
    def update_compressed(self, k, v, i):
        """Store K/V and return reconstructed V for attention."""
        sk, pq, proj = self._get_resources(i, k.shape[-1], k.device)
        self._allocate_buffers(i, k.shape[0], k.shape[1], k.shape[-1], k.device, initial_len=k.shape[2])
        start = self._cur_len[i]; total = start + k.shape[2]
        
        # Keys
        kz, _, _ = self._extract_outliers(k, i); ksk = torch.matmul(kz, sk).contiguous()
        rn, pn = triton_polar_encode(ksk, pq.get_all_boundaries(device=k.device), k.shape[-1], bits=pq.bits)
        self._final_radii_buf[i][:, :, start:total, :] = rn
        for lv, b in enumerate(pn): self._packed_angles_buf[i][lv][:, :, start:total, :] = b
        
        # Residual correction (QJL)
        k_rs = triton_polar_decode(rn, pn, pq.get_all_centroids(device=k.device), k.shape[-1], bits=pq.bits)
        pqjl, g_n = self._compute_qjl(ksk, k_rs, proj.to(k.device))
        self._packed_qjl_buf[i][:, :, start:total, :] = pqjl
        self._qjl_gammas_buf[i][:, :, start:total, :] = g_n
        
        # Values
        vn, vst = self._value_quantizer.quantize(v); self._values_buf[i][:, :, start:total, :] = vn
        if vst is not None: self._value_states_buf[i][:, :, start:total, :] = vst
        self._cur_len[i] = total
        return self._value_quantizer.dequantize(vn, vst, k.dtype)

    def fused_scores(self, q, i):
        """Compute attention scores directly on packed polar data (Elite V3)."""
        T = self._cur_len[i]; sk = self._sketch_matrices[i]; D = sk.shape[0]
        _, pq, proj = self._get_resources(i, D, q.device)
        qz, _, _ = self._extract_outliers(q, i); qsk = torch.matmul(qz, sk).contiguous()
        
        # Reconstruction-based for now (but bit-accurate with V3 kernels)
        k_rs = self._reconstruct_keys(i, T)
        # Apply score computation
        return torch.matmul(qsk, torch.matmul(k_rs, sk).transpose(-1, -2))


    def _compress_layer(self, i, k_new, v_new):
        raw_k = torch.cat([self._raw_keys.get(i, torch.empty((k_new.shape[0], k_new.shape[1], 0, k_new.shape[3]), device=k_new.device, dtype=k_new.dtype)), k_new], dim=2)
        raw_v = torch.cat([self._raw_values.get(i, torch.empty((v_new.shape[0], v_new.shape[1], 0, v_new.shape[3]), device=v_new.device, dtype=v_new.dtype)), v_new], dim=2)
        B, H, T, D = raw_k.shape; sk, pq, proj = self._get_resources(i, D, raw_k.device); self._allocate_buffers(i, B, H, D, raw_k.device)
        k_z, _, _ = self._extract_outliers(raw_k, i)
        k_sk = torch.matmul(k_z, sk).contiguous()
        print(f"DEBUG[Cache] Compress Layer {i} pq.bits={pq.bits} D={D}", flush=True)
        if is_triton_available() and raw_k.is_cuda:
            rf, pa = triton_polar_encode(k_sk, pq.get_all_boundaries(device=raw_k.device), D, bits=pq.bits)
            k_rs = triton_polar_decode(rf, pa, pq.get_all_centroids(device=raw_k.device), D, bits=pq.bits)
        else:
            rf, angs = recursive_polar_transform(k_sk); idx = pq.quantize_all(angs); pa = pq.pack_all(idx)
            unp = pq.unpack_all(pa); dec = pq.dequantize_all(unp); k_rs = recursive_polar_inverse(rf, dec)

        p_qjl, g = self._compute_qjl(k_sk, k_rs, proj.to(k_sk.device))
        self._final_radii_buf[i][:, :, :T, :] = rf.view(B, H, T, 1)
        for lv in range(len(pa)):
            self._packed_angles_buf[i][lv][:, :, :T, :] = pa[lv].view(B, H, T, -1)
        self._packed_qjl_buf[i][:, :, :T, :] = p_qjl.view(B, H, T, -1); self._qjl_gammas_buf[i][:, :, :T, :] = g.view(B, H, T, 1)
        # Values
        vn, vst = self._value_quantizer.quantize(raw_v); self._values_buf[i][:, :, :T, :] = vn
        if vst is not None: self._value_states_buf[i][:, :, :T, :] = vst
        self._cur_len[i] = T; self._compressed[i] = True; self._raw_keys.pop(i, None); self._raw_values.pop(i, None)

    def _compute_qjl(self, k_sk, k_rs, proj):
        u = torch.matmul(k_sk - k_rs, proj.to(k_sk.device))
        s = torch.sign(u); s = torch.where(s==0, torch.ones_like(s), s)
        from .bitpack import pack_1bit; return pack_1bit(s.to(torch.int8)), torch.abs(u).mean(dim=-1, keepdim=True)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        B, H, T_new, D = key_states.shape
        if self.dtype is None: self.dtype = key_states.dtype
        sk, pq, proj = self._get_resources(layer_idx, D, key_states.device)
        if layer_idx not in self._final_radii_buf: self._allocate_buffers(layer_idx, B, H, D, key_states.device, initial_len=T_new)
        else: self._ensure_capacity(layer_idx, self._cur_len[layer_idx] + T_new)

        if layer_idx == 0: self._seen_tokens += T_new
        if not self._compressed.get(layer_idx):
            if self._seen_tokens < self.compress_start:
                self._raw_keys[layer_idx] = torch.cat([self._raw_keys.get(layer_idx, torch.empty((B, H, 0, D), device=key_states.device, dtype=self.dtype)), key_states], dim=2)
                self._raw_values[layer_idx] = torch.cat([self._raw_values.get(layer_idx, torch.empty((B, H, 0, value_states.shape[-1]), device=value_states.device, dtype=self.dtype)), value_states], dim=2)
                return self._raw_keys[layer_idx], self._raw_values[layer_idx]
            else: self._compress_layer(layer_idx, key_states, value_states)
        else: self._update_internal(layer_idx, key_states, value_states)

        T = self._cur_len[layer_idx]
        k_full = self._reconstruct_keys(layer_idx, T); k_full = self._inject_outliers(k_full, layer_idx)
        v_full = self._value_quantizer.dequantize(self._values_buf[layer_idx][:, :, :T, :], self._value_states_buf.get(layer_idx)[:, :, :T, :] if layer_idx in self._value_states_buf else None, self.dtype)
        return k_full, v_full

    def _update_internal(self, i, k_n, v_n):
        B, H, T_n, D = k_n.shape; sk, pq, proj = self._get_resources(i, D, k_n.device)
        start = self._cur_len[i]; total = start + T_n
        kz, _, _ = self._extract_outliers(k_n, i); ksk = torch.matmul(kz, sk).contiguous()
        if is_triton_available() and k_n.is_cuda:
            rn, pn = triton_polar_encode(ksk, pq.get_all_boundaries(device=k_n.device), D, bits=pq.bits)
            krsn = triton_polar_decode(rn, pn, pq.get_all_centroids(device=k_n.device), D, bits=pq.bits)
        else:
            rn, an = recursive_polar_transform(ksk); idx = pq.quantize_all(an); pn = pq.pack_all(idx)
            unp = pq.unpack_all(pn); dec = pq.dequantize_all(unp); krsn = recursive_polar_inverse(rn, dec)
        pqjl, g_n = self._compute_qjl(ksk, krsn, proj.to(ksk.device))
        self._final_radii_buf[i][:, :, start:total, :] = rn
        for lv in range(len(pn)): self._packed_angles_buf[i][lv][:, :, start:total, :] = pn[lv]
        self._packed_qjl_buf[i][:, :, start:total, :] = pqjl; self._qjl_gammas_buf[i][:, :, start:total, :] = g_n
        vn, vst = self._value_quantizer.quantize(v_n); self._values_buf[i][:, :, start:total, :] = vn
        if vst is not None: self._value_states_buf[i][:, :, start:total, :] = vst
        self._cur_len[i] = total

    def _reconstruct_keys(self, i, T=None):
        if i not in self._final_radii_buf: return None
        if T is None: T = self._cur_len[i]
        sk = self._sketch_matrices[i]; D = sk.shape[0]; _, pq, proj = self._get_resources(i, D, self._final_radii_buf[i].device)
        rf = self._final_radii_buf[i][:, :, :T, :]; pa = [b[:, :, :T, :] for b in self._packed_angles_buf[i]]
        if is_triton_available() and rf.is_cuda: k_rs = triton_polar_decode(rf, pa, pq.get_all_centroids(device=rf.device), D, bits=pq.bits)
        else:
            unp = pq.unpack_all(pa); dec = pq.dequantize_all(unp); k_rs = recursive_polar_inverse(rf, dec)
        p_qjl = self._packed_qjl_buf[i][:, :, :T, :]; g = self._qjl_gammas_buf[i][:, :, :T, :]
        from .bitpack import unpack_1bit; qs = unpack_1bit(p_qjl, D).to(self.dtype)
        # Force proj to reconstruction device
        p_rec = proj.to(qs.device)
        corr = (qs @ p_rec.T) * (g * (math.sqrt(math.pi / 2) / D))
        return torch.matmul(k_rs + corr, sk.to(k_rs.device).T)

    def get_seq_length(self, i=0): return self._cur_len.get(i, 0)
    def get_mask_sizes(self, q_len, layer_idx=0): return self.get_seq_length(layer_idx) + (q_len.shape[0] if torch.is_tensor(q_len) else q_len), 0