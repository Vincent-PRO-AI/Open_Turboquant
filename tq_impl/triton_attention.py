import torch
import triton
import triton.language as tl
import math
from typing import List
from .triton_polar import is_triton_available, _TR_AVAIL

if _TR_AVAIL:
    from triton.language.extra import libdevice

    @triton.jit
    def _triton_fused_polar_attention_decode_kernel(
        Q_proj_ptr, Q_qjl_ptr, R_ptr, P_ptr, O_ptr, C_ptr, 
        Outlier_Idx_ptr, Outlier_Val_ptr, # 🚀 Outlier Injection
        QJL_P_ptr, QJL_G_ptr, Scores_ptr,
        B, H_q, H_kv, T_cache, D: tl.constexpr, L: tl.constexpr, bits: tl.constexpr,
        num_outliers: tl.constexpr,
        snqpb, snqph, snqpd,
        snqqb, snqqh, snqqd,
        snrb, snrh, snrt,
        snpb, snph, snpt,
        snov_b, snov_h, snov_t, # Outlier Val strides
        snqjlp_b, snqjlp_h, snqjlp_t,
        snqjlg_b, snqjlg_h, snqjlg_t,
        sn_scb, sn_sch, sn_sct
    ):
        pid_t = tl.program_id(0); pid_h = tl.program_id(1); pid_b = tl.program_id(2)
        if pid_t >= T_cache: return
        
        # GQA Mapping
        kv_h = pid_h // (H_q // H_kv)
        
        # Load Root R
        rf = tl.load(R_ptr + pid_b * snrb + kv_h * snrh + pid_t * snrt).to(tl.float32)
        
        # 🚀 Elite V3: Pure Register Polar Vector Reconstruction (Unrolled)
        iD = tl.arange(0, D)
        radii = tl.full([D], rf, dtype=tl.float32)
        p_token_base = P_ptr + pid_b * snpb + kv_h * snph + pid_t * snpt

        # Loop through expansion levels (Root to Leaves)
        for rev_lv in tl.static_range(L):
            lv = L - 1 - rev_lv
            half_block_depth = lv
            is_right = (iD >> half_block_depth) & 1
            ang_idx = iD >> (lv + 1)
            
            lvl_off = tl.load(O_ptr + lv)
            byte_off = lvl_off + (ang_idx * bits) // 8
            pb = tl.load(p_token_base + byte_off).to(tl.int32)
            
            bit_shift = (ang_idx * bits) % 8
            q_idx = (pb >> bit_shift) & (0x0F if bits == 4 else 0x07)
            
            phi = tl.load(C_ptr + lv * 16 + q_idx)
            factor = tl.where(is_right == 1, libdevice.sin(phi), libdevice.cos(phi))
            radii *= factor

        # 🚀 Outlier Injection (Register-Only)
        # Restore high-precision values for the top dynamic outliers
        for oi in tl.static_range(num_outliers):
            # Index of the pair (0..D/2-1)
            oidx = tl.load(Outlier_Idx_ptr + kv_h * num_outliers + oi).to(tl.int32)
            # Two values per pair
            v0 = tl.load(Outlier_Val_ptr + pid_b * snov_b + kv_h * snov_h + pid_t * snov_t + 2 * oi).to(tl.float32)
            v1 = tl.load(Outlier_Val_ptr + pid_b * snov_b + kv_h * snov_h + pid_t * snov_t + 2 * oi + 1).to(tl.float32)
            radii = tl.where(iD == 2 * oidx, v0, radii)
            radii = tl.where(iD == 2 * oidx + 1, v1, radii)

        # 🚀 Scoring
        mask_d = iD < D
        q_proj = tl.load(Q_proj_ptr + pid_b * snqpb + pid_h * snqph + iD * snqpd, mask=mask_d, other=0.0).to(tl.float32)
        q_qjl = tl.load(Q_qjl_ptr + pid_b * snqqb + pid_h * snqqh + iD * snqqd, mask=mask_d, other=0.0).to(tl.float32)
        
        score_base = tl.sum(q_proj * radii, axis=0)
        
        # QJL residual scoring (Uses robust strides)
        g_val = tl.load(QJL_G_ptr + pid_b * snqjlg_b + kv_h * snqjlg_h + pid_t * snqjlg_t).to(tl.float32)
        p_qjl = tl.load(QJL_P_ptr + pid_b * snqjlp_b + kv_h * snqjlp_h + pid_t * snqjlp_t + (iD // 8), mask=mask_d, other=0).to(tl.int32)
        bit_idx = iD % 8
        qs = ((p_qjl >> bit_idx) & 1).to(tl.float32) * 2.0 - 1.0
        score_qjl = tl.sum(q_qjl * qs, axis=0) * g_val
        
        # Store result
        tl.store(Scores_ptr + pid_b * sn_scb + pid_h * sn_sch + pid_t * sn_sct, (score_base + score_qjl).to(Scores_ptr.dtype.element_ty))

def triton_fused_polar_attention_decode(
    Q_proj: torch.Tensor, Q_qjl: torch.Tensor, R_out: torch.Tensor, P_flat: torch.Tensor, 
    offsets_t: torch.Tensor, centroids: torch.Tensor, 
    outlier_idx: torch.Tensor, outlier_vals: torch.Tensor, # 🚀
    p_qjl: torch.Tensor, g_val: torch.Tensor,
    D: int, bits: int
):
    if is_triton_available() and R_out.is_cuda:
        B, H_q, _, _ = Q_proj.shape
        _, H_kv, T_cache, _ = R_out.shape
        L = int(math.log2(D))
        dev = R_out.device; dtype = R_out.dtype
        num_outliers = outlier_idx.shape[1]
        
        num_outliers = outlier_idx.shape[1]
        
        Scores_out = torch.empty((B, H_q, 1, T_cache), device=dev, dtype=dtype)
        
        with torch.cuda.device(dev):
            _triton_fused_polar_attention_decode_kernel[(T_cache, H_q, B)](
                Q_proj, Q_qjl, R_out, P_flat, offsets_t, centroids,
                outlier_idx, outlier_vals,
                p_qjl, g_val, Scores_out,
                B, H_q, H_kv, T_cache, int(D), int(L), int(bits),
                int(num_outliers),
                Q_proj.stride(0), Q_proj.stride(1), Q_proj.stride(3),
                Q_qjl.stride(0), Q_qjl.stride(1), Q_qjl.stride(3),
                R_out.stride(0), R_out.stride(1), R_out.stride(2),
                P_flat.stride(0), P_flat.stride(1), P_flat.stride(2),
                outlier_vals.stride(0), outlier_vals.stride(1), outlier_vals.stride(2),
                p_qjl.stride(0), p_qjl.stride(1), p_qjl.stride(2),
                g_val.stride(0), g_val.stride(1), g_val.stride(2),
                Scores_out.stride(0), Scores_out.stride(1), Scores_out.stride(3),
                num_warps=4
            )
        return Scores_out
    else:
        raise RuntimeError("Triton unavailable, fused attention decode failed.")
