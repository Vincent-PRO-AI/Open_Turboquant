"""
tq_impl/triton_polar.py — Triton kernels for PolarQuant encode/decode
=====================================================================

Fused Triton kernels for the recursive polar transformation used in
PolarQuant (AISTATS 2026). Optimized for head_dim=128/256 and BFloat16.
"""
import torch
import math
from typing import Optional, List

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice
    _TR_AVAIL = True
except ImportError:
    _TR_AVAIL = False

def is_triton_available():
    return _TR_AVAIL and torch.cuda.is_available()

def triton_version():
    if not _TR_AVAIL: return "N/A"
    return triton.__version__


if _TR_AVAIL:

    @triton.jit
    def _triton_polar_encode_kernel(
        X_ptr, R_out_ptr, P_base_ptr, P_offsets_ptr, B_ptr, Scratch_ptr,
        B, H, T, D: tl.constexpr, L: tl.constexpr,
        stride_xb, stride_xh, stride_xt, stride_xd,
        stride_rb, stride_rh, stride_rt,
        stride_s,
    ):
        pid_t = tl.program_id(0); pid_h = tl.program_id(1); pid_b = tl.program_id(2)
        if pid_t >= T: return
        
        # DRAM Scratchpad Base (8192 float32 slots per token to be extra safe)
        s_base = Scratch_ptr + (pid_b * H * T + pid_h * T + pid_t) * 8192
        x_base = X_ptr + pid_b * stride_xb + pid_h * stride_xh + pid_t * stride_xt

        o256 = tl.arange(0, 256)
        xv = tl.load(x_base + o256, mask=o256 < D, other=0.0).to(tl.float32)
        tl.store(s_base + o256, xv, mask=o256 < D)

        for lv in tl.static_range(L):
            n_p = D >> (lv + 1)
            k = tl.arange(0, 128)

            r_o = lv * 256
            w_o = (lv + 1) * 256

            # Ensure radii from previous level are visible (barrier not needed with num_warps=1 but good practice)
            # Actually Triton DRAM access is global-memory consistent within a block if sequential.
            xi = tl.load(s_base + r_o + 2 * k, mask=k < n_p, other=0.0)
            yi = tl.load(s_base + r_o + 2 * k + 1, mask=k < n_p, other=0.0)

            ri = tl.sqrt(xi * xi + yi * yi + 1e-6)
            phi = libdevice.atan2(yi, xi)
            phi = tl.where(phi < 0, phi + 6.283185307, phi)

            bits = 4 if lv <= 3 else 2
            idx = tl.zeros([128], dtype=tl.int32)
            n_b = (1 << bits) - 1
            for bi in tl.static_range(15):
                bd = tl.load(B_ptr + lv * 16 + bi)
                idx = tl.where((phi > bd + 1e-9) & (k < n_p), bi + 1, idx)
            idx = tl.where(idx > n_b, n_b, idx)

            idx_base = 4096 + lv * 128
            tl.store(s_base + idx_base + k, idx, mask=k < n_p)

            # Pack
            pos_offset = (pid_b * H * T + pid_h * T + pid_t)
            offset_val = tl.load(P_offsets_ptr + lv)
            if bits == 4:
                ppp4 = n_p // 2 if n_p >= 2 else 1
                p_ptr_4 = P_base_ptr + offset_val + pos_offset * ppp4
                k64 = tl.arange(0, 64)
                m64 = k64 < ppp4
                vd0 = tl.load(s_base + idx_base + 2 * k64, mask=(2*k64 < n_p), other=0).to(tl.int32)
                vd1 = tl.load(s_base + idx_base + 2 * k64 + 1, mask=(2*k64+1 < n_p), other=0).to(tl.int32)
                tl.store(p_ptr_4 + k64, (vd0 | (vd1 << 4)).to(tl.uint8), mask=m64)
            else:
                ppp2 = n_p // 4 if n_p >= 4 else 1
                p_ptr_2 = P_base_ptr + offset_val + pos_offset * ppp2
                k32 = tl.arange(0, 32)
                m32 = k32 < ppp2
                ve0 = tl.load(s_base + idx_base + 4 * k32, mask=(4*k32 < n_p), other=0).to(tl.int32)
                ve1 = tl.load(s_base + idx_base + 4 * k32 + 1, mask=(4*k32+1 < n_p), other=0).to(tl.int32)
                ve2 = tl.load(s_base + idx_base + 4 * k32 + 2, mask=(4*k32+2 < n_p), other=0).to(tl.int32)
                ve3 = tl.load(s_base + idx_base + 4 * k32 + 3, mask=(4*k32+3 < n_p), other=0).to(tl.int32)
                tl.store(p_ptr_2 + k32, (ve0 | (ve1 << 2) | (ve2 << 4) | (ve3 << 6)).to(tl.uint8), mask=m32)

            tl.store(s_base + w_o + k, ri, mask=k < n_p)

        tl.store(
            R_out_ptr + pid_b * stride_rb + pid_h * stride_rh + pid_t * stride_rt,
            tl.load(s_base + L * 256).to(R_out_ptr.dtype.element_ty),
        )

    @triton.jit
    def _triton_polar_decode_kernel(
        R_ptr, P_base_ptr, P_offsets_ptr, C_ptr, K_out_ptr, Scratch_ptr,
        B, H, T, D: tl.constexpr, L: tl.constexpr,
        stride_rb, stride_rh, stride_rt,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_s,
    ):
        pid_t = tl.program_id(0); pid_h = tl.program_id(1); pid_b = tl.program_id(2)
        if pid_t >= T: return
        s_base = Scratch_ptr + (pid_b * H * T + pid_h * T + pid_t) * 8192

        r_val = tl.load(R_ptr + pid_b * stride_rb + pid_h * stride_rh + pid_t * stride_rt).to(tl.float32)
        tl.store(s_base + L * 256, r_val)

        for rev_lv in tl.static_range(L):
            lv = L - 1 - rev_lv
            n_p = D >> (lv + 1)
            k = tl.arange(0, 128)

            bits = 4 if lv <= 3 else 2
            idx_base = 4096 + lv * 128
            pos_offset = (pid_b * H * T + pid_h * T + pid_t)
            offset_val = tl.load(P_offsets_ptr + lv)
            
            if bits == 4:
                ppp4 = n_p // 2 if n_p >= 2 else 1
                p_ptr_4 = P_base_ptr + offset_val + pos_offset * ppp4
                k64 = tl.arange(0, 64)
                m64 = k64 < ppp4
                pb4 = tl.load(p_ptr_4 + k64, mask=m64, other=0).to(tl.int32)
                tl.store(s_base + idx_base + 2 * k64, pb4 & 0x0F, mask=(2*k64 < n_p))
                tl.store(s_base + idx_base + 2 * k64 + 1, (pb4 >> 4) & 0x0F, mask=(2*k64+1 < n_p))
            else:
                ppp2 = n_p // 4 if n_p >= 4 else 1
                p_ptr_2 = P_base_ptr + offset_val + pos_offset * ppp2
                k32 = tl.arange(0, 32)
                m32 = k32 < ppp2
                pb2 = tl.load(p_ptr_2 + k32, mask=m32, other=0).to(tl.int32)
                tl.store(s_base + idx_base + 4 * k32, pb2 & 0x03, mask=(4*k32 < n_p))
                tl.store(s_base + idx_base + 4 * k32 + 1, (pb2 >> 2) & 0x03, mask=(4*k32+1 < n_p))
                tl.store(s_base + idx_base + 4 * k32 + 2, (pb2 >> 4) & 0x03, mask=(4*k32+2 < n_p))
                tl.store(s_base + idx_base + 4 * k32 + 3, (pb2 >> 6) & 0x03, mask=(4*k32+3 < n_p))

            r_o = (lv + 1) * 256
            w_o = lv * 256
            ri = tl.load(s_base + r_o + k, mask=k < n_p, other=0.0)
            idx = tl.load(s_base + idx_base + k, mask=k < n_p, other=0).to(tl.int32)
            phi = tl.load(C_ptr + lv * 16 + idx)
            
            tl.store(s_base + w_o + 2 * k, ri * tl.cos(phi), mask=k < n_p)
            tl.store(s_base + w_o + 2 * k + 1, ri * tl.sin(phi), mask=k < n_p)

        o256 = tl.arange(0, 256)
        k_out_base = K_out_ptr + pid_b * stride_kb + pid_h * stride_kh + pid_t * stride_kt
        tl.store(k_out_base + o256, tl.load(s_base + o256, mask=o256 < D).to(K_out_ptr.dtype.element_ty), mask=o256 < D)


    def triton_polar_encode(k_sk: torch.Tensor, boundaries: torch.Tensor, D: int):
        if not (_TR_AVAIL and k_sk.is_cuda):
            from .polar import recursive_polar_transform
            from .polar_quant import PolarAngleQuantizer
            pq = PolarAngleQuantizer(d=D)
            rf, angs = recursive_polar_transform(k_sk); idx = pq.quantize_all(angs); pa = pq.pack_all(idx)
            return rf, pa

        B, H, T, _ = k_sk.shape; L = int(math.log2(D))
        bd_flat = boundaries.to(k_sk.device).contiguous().view(-1).to(torch.float32)
        offsets = [0]
        for lv in range(L):
            n_p = D >> (lv + 1); bits = 4 if lv <= 3 else 2
            ppp = max(1, (n_p * bits) // 8); offsets.append(offsets[-1] + B * H * T * ppp)
        offsets_t = torch.tensor(offsets[:-1], dtype=torch.int64, device=k_sk.device)
        R_out = torch.empty(B, H, T, 1, device=k_sk.device, dtype=k_sk.dtype)
        P_base = torch.empty(offsets[-1], device=k_sk.device, dtype=torch.uint8)
        scratch = torch.empty(B * H * T * 8192, device=k_sk.device, dtype=torch.float32)
        
        with torch.cuda.device(k_sk.device):
            _triton_polar_encode_kernel[(T, H, B)](
                k_sk, R_out, P_base, offsets_t, bd_flat, scratch,
                B, H, T, D, L,
                k_sk.stride(0), k_sk.stride(1), k_sk.stride(2), k_sk.stride(3),
                R_out.stride(0), R_out.stride(1), R_out.stride(2),
                8192,
                num_warps=1
            )
        
        p_a = []
        for lv in range(L):
            n_p = D >> (lv+1); bits = 4 if lv <= 3 else 2; ppp = max(1, (n_p*bits)//8)
            p_a.append(P_base[offsets[lv]:offsets[lv+1]].view(B, H, T, ppp))
        return R_out, p_a

    def triton_polar_decode(final_radii: torch.Tensor, packed_angles: list, centroids: torch.Tensor, D: int) -> torch.Tensor:
        if not (_TR_AVAIL and final_radii.is_cuda):
            from .polar import recursive_polar_inverse
            from .polar_quant import PolarAngleQuantizer
            pq = PolarAngleQuantizer(d=D); unpacked = pq.unpack_all(packed_angles); rec_angs = pq.dequantize_all(unpacked)
            return recursive_polar_inverse(final_radii, rec_angs)

        B, H, T, _ = final_radii.shape; L = int(math.log2(D))
        ct_flat = centroids.to(final_radii.device).contiguous().to(torch.float32)
        offsets = [0]
        for lv in range(L):
            n_p = D >> (lv+1); bits = 4 if lv <= 3 else 2
            ppp = max(1, (n_p*bits)//8); offsets.append(offsets[-1] + B * H * T * ppp)
        
        offsets_t = torch.tensor(offsets[:-1], dtype=torch.int64, device=final_radii.device)
        P_base = torch.empty(offsets[-1], device=final_radii.device, dtype=torch.uint8)
        for lv, pa in enumerate(packed_angles):
            P_base[offsets[lv]:offsets[lv+1]] = pa.to(final_radii.device).reshape(-1)
            
        K_out = torch.empty(B, H, T, D, device=final_radii.device, dtype=final_radii.dtype)
        scratch = torch.empty(B * H * T * 8192, device=final_radii.device, dtype=torch.float32)
        
        with torch.cuda.device(final_radii.device):
            _triton_polar_decode_kernel[(T, H, B)](
                final_radii, P_base, offsets_t, ct_flat, K_out, scratch,
                B, H, T, D, L,
                final_radii.stride(0), final_radii.stride(1), final_radii.stride(2),
                K_out.stride(0), K_out.stride(1), K_out.stride(2), K_out.stride(3),
                8192,
                num_warps=1
            )
        return K_out
else:
    def triton_polar_encode(*args, **kwargs): raise RuntimeError("Triton unavailable")
    def triton_polar_decode(*args, **kwargs): raise RuntimeError("Triton unavailable")
