"""
tq_impl/bitpack.py
------------------
Bit-level packing/unpacking for TurboQuant compressed keys.

Storage formats
---------------
2-bit MSE indices (4 per uint8):
    byte = idx3<<6 | idx2<<4 | idx1<<2 | idx0
    → D=128 → 32 bytes/position  (vs 256 bytes fp16 = 8x keys)

3-bit MSE indices (2 per uint8, 2 bits unused):
    byte = idx1<<3 | idx0
    → D=128 → 64 bytes/position  (vs 256 bytes fp16 = 4x keys)

1-bit QJL signs (8 per uint8):
    byte = b7<<7 | b6<<6 | ... | b1<<1 | b0
    where bi = 1 if sign=+1, 0 if sign=-1
    → D=128 → 16 bytes/position

All operations are pure PyTorch (GPU-compatible, differentiable-safe).
"""
from __future__ import annotations

import torch


# =====================================================================
# 2-bit packing  (for MSE with bits_mse=2, 4 centroids)
# =====================================================================

def pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack 2-bit indices (values 0–3) into uint8, 4 per byte.

    Input:  [..., D] int16/int32 with values in [0, 3]
    Output: [..., D//4] uint8
    """
    *lead, D = indices.shape
    assert D % 4 == 0, f"head_dim must be divisible by 4, got {D}"
    x = indices.reshape(*lead, D // 4, 4).to(torch.uint8)
    packed = x[..., 0] | (x[..., 1] << 2) | (x[..., 2] << 4) | (x[..., 3] << 6)
    return packed   # [..., D//4] uint8


def unpack_2bit(packed: torch.Tensor, D: int) -> torch.Tensor:
    """
    Unpack uint8 → 2-bit indices.
    """
    *lead, packed_D = packed.shape
    x0 =  packed       & 0x03
    x1 = (packed >> 2) & 0x03
    x2 = (packed >> 4) & 0x03
    x3 = (packed >> 6) & 0x03
    return torch.stack([x0, x1, x2, x3], dim=-1).reshape(*lead, D).to(torch.int16)


# =====================================================================
# 4-bit packing (for MSE or Polar Level 0)
# =====================================================================

def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack 4-bit indices (values 0–15) into uint8, 2 per byte.
    """
    *lead, D = indices.shape
    assert D % 2 == 0, f"head_dim must be even, got {D}"
    x = indices.reshape(*lead, D // 2, 2).to(torch.uint8)
    packed = x[..., 0] | (x[..., 1] << 4)
    return packed


def unpack_4bit(packed: torch.Tensor, D: int) -> torch.Tensor:
    """
    Unpack uint8 → 4-bit indices.
    """
    *lead, packed_D = packed.shape
    x0 =  packed       & 0x0F
    x1 = (packed >> 4) & 0x0F
    return torch.stack([x0, x1], dim=-1).reshape(*lead, D).to(torch.int16)


# =====================================================================
# 3-bit packing  (for MSE with bits_mse=3, 8 centroids)
# =====================================================================

def pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack 3-bit indices (values 0–7) into uint8, 2 per byte.
    Uses 6 of 8 bits (2 bits wasted per byte for simplicity).

    Input:  [..., D] int16/int32 with values in [0, 7]
    Output: [..., D//2] uint8
    """
    *lead, D = indices.shape
    assert D % 2 == 0, f"head_dim must be even, got {D}"
    x = indices.reshape(*lead, D // 2, 2).to(torch.uint8)
    packed = x[..., 0] | (x[..., 1] << 3)
    return packed   # [..., D//2] uint8


def unpack_3bit(packed: torch.Tensor, D: int) -> torch.Tensor:
    """
    Unpack uint8 → 3-bit indices.

    Input:  [..., D//2] uint8
    Output: [..., D] int16
    """
    *lead, packed_D = packed.shape
    x0 =  packed       & 0x07
    x1 = (packed >> 3) & 0x07
    return torch.stack([x0, x1], dim=-1).reshape(*lead, D).to(torch.int16)


# =====================================================================
# 1-bit packing  (for QJL signs)
# =====================================================================

def pack_1bit(signs: torch.Tensor) -> torch.Tensor:
    """
    Pack sign tensor ({-1, +1} as int8) into uint8, 8 per byte.

    Input:  [..., D] int8 with values in {-1, +1}
    Output: [..., D//8] uint8
    """
    *lead, D = signs.shape
    assert D % 8 == 0, f"head_dim must be divisible by 8, got {D}"
    # Convert {-1,+1} → {0,1}
    bits = ((signs.to(torch.int16) + 1) >> 1).to(torch.uint8)   # {-1→0, +1→1}
    bits = bits.reshape(*lead, D // 8, 8)
    packed = (
        bits[..., 0]       | (bits[..., 1] << 1) |
        (bits[..., 2] << 2) | (bits[..., 3] << 3) |
        (bits[..., 4] << 4) | (bits[..., 5] << 5) |
        (bits[..., 6] << 6) | (bits[..., 7] << 7)
    )
    return packed   # [..., D//8] uint8


def unpack_1bit(packed: torch.Tensor, D: int) -> torch.Tensor:
    """
    Unpack uint8 → 1-bit signs as float {-1.0, +1.0}.

    Input:  [..., D//8] uint8
    Output: [..., D] float16
    """
    *lead, packed_D = packed.shape
    bits = []
    for i in range(8):
        bits.append((packed >> i) & 1)
    bits_tensor = torch.stack(bits, dim=-1)    # [..., D//8, 8] uint8
    # {0, 1} → {-1.0, +1.0}
    return (bits_tensor.to(torch.float16) * 2.0 - 1.0).reshape(*lead, D)


# =====================================================================
# Memory accounting
# =====================================================================

def packed_bytes_per_position(bits_mse: int, head_dim: int) -> int:
    """
    Return actual bytes per (head, position) for packed TurboQuant keys.

    Components:
      - Packed MSE indices: D // pack_factor bytes
      - Packed QJL signs:   D // 8 bytes
      - Residual norm:      2 bytes (fp16)
      - Key norm:           2 bytes (fp16)
    """
    D = head_dim
    if bits_mse == 2:
        idx_bytes = D // 4      # 4 values per byte
    elif bits_mse == 3:
        idx_bytes = D // 2      # 2 values per byte (6-bit used)
    else:
        idx_bytes = D           # 1 value per byte (fallback)
    qjl_bytes = D // 8          # 8 signs per byte
    return idx_bytes + qjl_bytes + 2 + 2   # +2 each for res_norm, key_norm


def compression_ratio(bits_mse: int, head_dim: int) -> float:
    """
    Return compression ratio for keys vs FP16 baseline.

    FP16 baseline: head_dim * 2 bytes per position.
    """
    fp16_bytes = head_dim * 2
    tq_bytes = packed_bytes_per_position(bits_mse, head_dim)
    return fp16_bytes / tq_bytes