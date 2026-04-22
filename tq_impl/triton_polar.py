import torch
import triton
import triton.language as tl
import math
from typing import List, Optional

try:
    from triton.language.extra import libdevice
    _TR_AVAIL = True
except ImportError:
    _TR_AVAIL = False

triton_version = triton.__version__ if _TR_AVAIL else "N/A"

def is_triton_available():
    return _TR_AVAIL

if _TR_AVAIL:
    @triton.jit
    def _triton_polar_encode_kernel_v3(
        X_ptr, R_ptr, P_ptr, O_ptr, B_ptr, S_ptr,
        B, H, T, D: tl.constexpr, L: tl.constexpr, bits: tl.constexpr,
        snxb, snxh, snxt, snxd,
        snrb, snrh, snrt
    ):
        pid_t = tl.program_id(0).to(tl.int64); pid_h = tl.program_id(1).to(tl.int64); pid_b = tl.program_id(2).to(tl.int64)
        if pid_t >= T: return
        
        # 🚀 Fix: Safe 16KB Stride Alignment
        idx_64 = (pid_b * H * T + pid_h * T + pid_t)
        s_base = S_ptr + idx_64 * 16384
        x_base = X_ptr + pid_b * snxb + pid_h * snxh + pid_t * snxt
        
        PI = 3.14159265358979323846
        EPS = 1e-12
        
        # Load Level 0
        o256 = tl.arange(0, 256)
        tl.store(s_base + o256, tl.load(x_base + o256 * snxd, mask=o256 < D, other=0.0).to(tl.float32), mask=o256 < 256)
        tl.debug_barrier()
        
        for lv in tl.static_range(L):
            n_pairs = D >> (lv + 1)
            r_offset = lv * 256
            w_offset = (lv + 1) * 256
            idx_offset = 8192 + lv * 128
            
            k = tl.arange(0, 128)
            mask = k < n_pairs
            
            x = tl.load(s_base + r_offset + 2 * k, mask=mask, other=0.0)
            y = tl.load(s_base + r_offset + 2 * k + 1, mask=mask, other=0.0)
            
            ri = tl.sqrt(x * x + y * y + EPS)
            phi = libdevice.atan2(y, x)
            phi = tl.where(phi < 0.0, phi + 2.0 * PI, phi)
            
            tl.store(s_base + w_offset + k, ri, mask=mask)
            
            # Quantize
            idx = tl.zeros([128], dtype=tl.int32)
            for bi in tl.static_range(16):
                bd = tl.load(B_ptr + lv * 16 + bi)
                idx = tl.where((phi > bd + 1e-9) & mask, bi + 1, idx)
            idx = tl.where(idx >= (1 << bits), (1 << bits) - 1, idx)
            tl.store(s_base + idx_offset + k, idx.to(tl.float32), mask=mask)
            tl.debug_barrier()
            
            # Pack
            n_pairs_64 = n_pairs.to(tl.int64)
            p_offs = tl.load(O_ptr + lv).to(tl.int64) + idx_64 * (max(1, (n_pairs_64 * int(bits)) // 8))
            k64 = tl.arange(0, 64)
            m_pack = k64 < (max(1, n_pairs // 2))
            v0 = tl.load(s_base + idx_offset + 2 * k64, mask=(2*k64 < n_pairs), other=0).to(tl.int32)
            v1 = tl.load(s_base + idx_offset + 2 * k64 + 1, mask=(2*k64+1 < n_pairs), other=0).to(tl.int32)
            
            if bits == 4:
                packed = (v0 & 0x0F) | ((v1 & 0x0F) << 4)
                tl.store(P_ptr + p_offs + k64, packed.to(tl.uint8), mask=m_pack)
            else:
                packed = (v0 & 0x07) | ((v1 & 0x07) << 3)
                tl.store(P_ptr + p_offs + k64, packed.to(tl.uint8), mask=m_pack)
            tl.debug_barrier()
            
        rf = tl.load(s_base + L * 256).to(R_ptr.dtype.element_ty)
        tl.store(R_ptr + pid_b * snrb + pid_h * snrh + pid_t * snrt, rf)

    @triton.jit
    def _triton_polar_decode_kernel_v3(
        R_ptr, P_ptr, O_ptr, C_ptr, K_ptr, S_ptr,
        B, H, T, D: tl.constexpr, L: tl.constexpr, bits: tl.constexpr,
        snrb, snrh, snrt,
        snkb, snkh, snkt, snkd
    ):
        pid_t = tl.program_id(0).to(tl.int64); pid_h = tl.program_id(1).to(tl.int64); pid_b = tl.program_id(2).to(tl.int64)
        if pid_t >= T: return
        
        idx_64 = (pid_b * H * T + pid_h * T + pid_t)
        s_base = S_ptr + idx_64 * 16384
        
        rf = tl.load(R_ptr + pid_b * snrb + pid_h * snrh + pid_t * snrt).to(tl.float32)
        tl.store(s_base + L * 256, rf)
        tl.debug_barrier()

        for rev_lv in tl.static_range(L):
            lv = L - 1 - rev_lv
            n_pairs = D >> (lv + 1)
            r_offset = (lv + 1) * 256
            w_offset = lv * 256
            idx_offset = 8192 + lv * 128
            
            n_pairs_64 = n_pairs.to(tl.int64)
            p_offs = tl.load(O_ptr + lv).to(tl.int64) + idx_64 * (max(1, (n_pairs_64 * int(bits)) // 8))
            k64 = tl.arange(0, 64)
            m_pack = k64 < (max(1, n_pairs // 2))
            pb = tl.load(P_ptr + p_offs + k64, mask=m_pack, other=0).to(tl.int32)
            
            if bits == 4:
                tl.store(s_base + idx_offset + 2 * k64, (pb & 0x0F).to(tl.float32), mask=(2*k64 < n_pairs))
                tl.store(s_base + idx_offset + 2 * k64 + 1, ((pb >> 4) & 0x0F).to(tl.float32), mask=(2*k64+1 < n_pairs))
            else:
                tl.store(s_base + idx_offset + 2 * k64, (pb & 0x07).to(tl.float32), mask=(2*k64 < n_pairs))
                tl.store(s_base + idx_offset + 2 * k64 + 1, ((pb >> 3) & 0x07).to(tl.float32), mask=(2*k64+1 < n_pairs))
            tl.debug_barrier()
            
            k = tl.arange(0, 128)
            mask = k < n_pairs
            idx = tl.load(s_base + idx_offset + k, mask=mask, other=0).to(tl.int32)
            phi = tl.load(C_ptr + lv * 16 + idx, mask=mask, other=0.0)
            ri = tl.load(s_base + r_offset + k, mask=mask, other=0.0)
            
            x_rec = ri * libdevice.cos(phi)
            y_rec = ri * libdevice.sin(phi)
            
            tl.store(s_base + w_offset + 2 * k, x_rec, mask=mask)
            tl.store(s_base + w_offset + 2 * k + 1, y_rec, mask=mask)
            tl.debug_barrier()

        k_out_base = K_ptr + pid_b * snkb + pid_h * snkh + pid_t * snkt
        o256 = tl.arange(0, 256)
        final_vals = tl.load(s_base + o256, mask=o256 < D).to(K_ptr.dtype.element_ty)
        tl.store(k_out_base + o256 * snkd, final_vals, mask=o256 < D)

    def triton_polar_encode(k_sk: torch.Tensor, boundaries: torch.Tensor, D: int, bits: int, scratch: Optional[torch.Tensor] = None):
        if is_triton_available() and k_sk.is_cuda:
            B, H, T, _ = k_sk.shape; L = int(math.log2(D)); dev = k_sk.device; dtype = k_sk.dtype
            k_sk = k_sk.contiguous(); bd_flat = boundaries.to(dev).contiguous()
            
            # Pack offsets calculation
            offsets = [0]
            for lv in range(L):
                n_p = D >> (lv+1); ppp = max(1, (n_p * int(bits)) // 8); offsets.append(offsets[-1] + B * H * T * ppp)
            offsets_t = torch.tensor(offsets[:-1], dtype=torch.int64, device=dev)
            
            R_out = torch.empty(B, H, T, 1, device=dev, dtype=dtype)
            P_base = torch.empty(offsets[-1], device=dev, dtype=torch.uint8)
            
            # Use provided scratch or allocate a temporary one
            if scratch is None:
                scratch = torch.empty(B * H * T * 16384, device=dev, dtype=torch.float32)
            
            with torch.cuda.device(dev):
                _triton_polar_encode_kernel_v3[(T, H, B)](
                    k_sk, R_out, P_base, offsets_t, bd_flat, scratch, 
                    B, H, T, int(D), int(L), int(bits), 
                    k_sk.stride(0), k_sk.stride(1), k_sk.stride(2), k_sk.stride(3), 
                    R_out.stride(0), R_out.stride(1), R_out.stride(2), 
                    num_warps=4
                )
            p_a = []
            for lv in range(L):
                n_p = D >> (lv+1); ppp = max(1, (n_p * int(bits)) // 8)
                p_a.append(P_base[offsets[lv]:offsets[lv+1]].view(B, H, T, ppp))
            return R_out, p_a
        else:
            from .polar import recursive_polar_transform
            from .polar_quant import PolarAngleQuantizer
            pq = PolarAngleQuantizer(d=k_sk.shape[-1], bits=int(bits))
            rf, angs = recursive_polar_transform(k_sk); idx = pq.quantize_all(angs); pa = pq.pack_all(idx)
            return rf, pa

    def triton_polar_decode(R_out: torch.Tensor, p_a: List[torch.Tensor], centroids: torch.Tensor, D: int, bits: int):
        if is_triton_available() and R_out.is_cuda:
            B, H, T, _ = R_out.shape; L = int(math.log2(D)); dev = R_out.device; dtype = R_out.dtype
            R_out = R_out.contiguous(); ct_flat = centroids.to(dev).contiguous()
            offsets = [0]
            for lv in range(L):
                n_p = D >> (lv+1); ppp = max(1, (n_p * int(bits)) // 8); offsets.append(offsets[-1] + B * H * T * ppp)
            offsets_t = torch.tensor(offsets[:-1], dtype=torch.int64, device=dev)
            P_base = torch.empty(offsets[-1], device=dev, dtype=torch.uint8)
            for lv, pa in enumerate(p_a): P_base[offsets[lv]:offsets[lv+1]] = pa.reshape(-1).to(dev).contiguous()
            K_out = torch.empty(B, H, T, D, device=dev, dtype=dtype)
            scratch = torch.empty(B * H * T * 16384, device=dev, dtype=torch.float32)
            with torch.cuda.device(dev):
                _triton_polar_decode_kernel_v3[(T, H, B)](
                    R_out, P_base, offsets_t, ct_flat, K_out, scratch, 
                    B, H, T, int(D), int(L), int(bits), 
                    R_out.stride(0), R_out.stride(1), R_out.stride(2), 
                    K_out.stride(0), K_out.stride(1), K_out.stride(2), K_out.stride(3), 
                    num_warps=4
                )
            return K_out
        else:
            from .polar_quant import PolarAngleQuantizer
            from .polar import recursive_polar_inverse
            pq = PolarAngleQuantizer(d=D, bits=int(bits)); unp = pq.unpack_all(p_a)
            dec = pq.dequantize_all(unp); return recursive_polar_inverse(R_out, dec)
else:
    def triton_polar_encode(*args, **kwargs): raise RuntimeError("Triton unavailable")
    def triton_polar_decode(*args, **kwargs): raise RuntimeError("Triton unavailable")
