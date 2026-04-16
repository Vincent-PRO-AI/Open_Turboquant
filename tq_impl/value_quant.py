import torch
from typing import Tuple, Optional
from .bitpack import pack_2bit, unpack_2bit # reuse if it supports D divisible by 4

def pack_4bit_value(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit indices into uint8 (2 per byte) for Values."""
    *lead, D = indices.shape
    x = indices.reshape(*lead, D // 2, 2).to(torch.uint8)
    return x[..., 0] | (x[..., 1] << 4)

def unpack_4bit_value(packed: torch.Tensor, D: int) -> torch.Tensor:
    """Unpack uint8 into 4-bit indices."""
    *lead, packed_D = packed.shape
    x0 = packed & 0x0F
    x1 = (packed >> 4) & 0x0F
    return torch.stack([x0, x1], dim=-1).reshape(*lead, D).to(torch.int16)

class ValueQuantizer:
    """
    Simple Quantizer for Values in KV Cache.
    Supports 8-bit (FP8) and 4-bit (INT4 per head).
    """
    def __init__(self, bits: int = 8, use_fp8: bool = True):
        self.bits = bits
        self.use_fp8 = use_fp8
        
    def quantize(self, v: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Input: [B, KVH, T, D] FP16
        Output: (Packed Tensor, Scales | None)
        """
        if self.bits >= 16:
            return v, None
            
        if self.bits == 8:
            if self.use_fp8 and hasattr(torch, 'float8_e4m3fn'):
                return v.to(torch.float8_e4m3fn), None
            else:
                # Fallback to int8 per-head
                scale = v.abs().max(dim=-1, keepdim=True).values / 127.0
                q = (v / scale.clamp(min=1e-6)).round().clamp(-128, 127).to(torch.int8)
                return q, scale
                
        if self.bits == 4:
            # Min-Max 4-bit per-head
            v_min = v.min(dim=-1, keepdim=True).values
            v_max = v.max(dim=-1, keepdim=True).values
            scale = (v_max - v_min).clamp(min=1e-6) / 15.0
            
            q = ((v - v_min) / scale).round().clamp(0, 15).to(torch.int16)
            packed = pack_4bit_value(q)
            # We pack (min, scale) into fp16
            return packed, torch.cat([v_min, scale], dim=-1)
            
        return v, None

    def dequantize(self, q: torch.Tensor, state: Optional[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        if self.bits >= 16:
            return q.to(dtype)
            
        if self.bits == 8:
            if self.use_fp8 and isinstance(q, torch.Tensor) and q.dtype == torch.float8_e4m3fn:
                return q.to(dtype)
            else:
                return (q.to(dtype) * state)
                
        if self.bits == 4:
            D = q.shape[-1] * 2
            indices = unpack_4bit_value(q, D)
            v_min = state[..., 0:1]
            scale = state[..., 1:2]
            return (indices.to(dtype) * scale + v_min)
            
        return q.to(dtype)
