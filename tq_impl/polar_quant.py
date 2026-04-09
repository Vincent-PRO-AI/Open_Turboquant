import torch
import math
from typing import List, Tuple
from .codebook import get_angular_codebook, get_angular_boundaries
from .bitpack import pack_4bit, unpack_4bit, pack_2bit, unpack_2bit, pack_3bit, unpack_3bit

class PolarAngleQuantizer:
    """
    Hierarchical Angle Quantizer for PolarQuant v2 (AISTATS 2026).
    Uses optimal non-uniform codebooks for the recursive angular distributions.
    """
    def __init__(self, d: int = 128):
        self.d = d
        self.n_levels = int(math.log2(d))

    def _get_bits(self, level: int) -> int:
        # Boost first 4 levels to 4 bits for maximum precision in the early tree
        if level <= 3: return 4
        return 2

    def quantize_level(self, phi: torch.Tensor, level: int) -> torch.Tensor:
        """Find nearest indices in the level's optimal codebook."""
        bits = self._get_bits(level)
        boundaries = get_angular_boundaries(bits, level).to(phi.device)
        indices = torch.bucketize(phi, boundaries)
        return torch.clamp(indices, 0, (2**bits) - 1).to(torch.uint8)

    def dequantize_level(self, indices: torch.Tensor, level: int) -> torch.Tensor:
        """Map indices back to optimal centroids."""
        bits = self._get_bits(level)
        cb = get_angular_codebook(bits, level).to(indices.device)
        return cb[indices.long()]

    def quantize_all(self, angles: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.quantize_level(phi, i) for i, phi in enumerate(angles)]

    def dequantize_all(self, indices_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.dequantize_level(idx, i) for i, idx in enumerate(indices_list)]

    def compute_qjl_residual(self, x: torch.Tensor, x_rec: torch.Tensor, proj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the 1-bit QJL correction for the quantization residual.
        Ensures unbiasedness of the inner products.
        """
        res = x - x_rec
        u = torch.matmul(res, proj)
        sign = torch.sign(u).to(torch.int8)
        gamma = torch.abs(u).mean(dim=-1, keepdim=True)
        return sign, gamma

    def pack_all(self, indices_list: List[torch.Tensor]) -> List[torch.Tensor]:
        packed = []
        for i, idx in enumerate(indices_list):
            bits = self._get_bits(i)
            level_d = idx.shape[-1]
            if bits == 4 and level_d % 2 == 0:
                packed.append(pack_4bit(idx))
            elif bits == 3 and level_d % 2 == 0:
                packed.append(pack_3bit(idx))
            elif bits == 2:
                if level_d >= 4:
                    packed.append(pack_2bit(idx))
                elif level_d == 2:
                    packed.append((idx[..., 0] | (idx[..., 1] << 2)).to(torch.uint8).unsqueeze(-1))
                elif level_d == 1:
                    packed.append((idx[..., 0] & 0x03).to(torch.uint8).unsqueeze(-1))
            else:
                packed.append(idx.to(torch.uint8))
        return packed

    def unpack_all(self, packed_list: List[torch.Tensor]) -> List[torch.Tensor]:
        unpacked = []
        for i, packed in enumerate(packed_list):
            bits = self._get_bits(i)
            # Recalculate original level_d
            level_d = self.d // (2**(i+1))
            if bits == 4 and level_d % 2 == 0:
                unpacked.append(unpack_4bit(packed, level_d))
            elif bits == 3 and level_d % 2 == 0:
                unpacked.append(unpack_3bit(packed, level_d))
            elif bits == 2:
                if level_d >= 4:
                    unpacked.append(unpack_2bit(packed, level_d))
                elif level_d == 2:
                    x0 = packed[..., 0] & 0x03
                    x1 = (packed[..., 0] >> 2) & 0x03
                    unpacked.append(torch.stack([x0, x1], dim=-1).to(torch.int16))
                elif level_d == 1:
                    unpacked.append((packed[..., 0] & 0x03).unsqueeze(-1).to(torch.int16))
            else:
                unpacked.append(packed.to(torch.int16))
        return unpacked

    # ------------------------------------------------------------------
    # Methods required by triton_polar / cache.py for Triton fast path
    # ------------------------------------------------------------------

    def get_all_boundaries(self) -> torch.Tensor:
        """
        Return a flat tensor of all level boundaries for Triton kernels.
        Shape: (n_levels, max_boundaries) padded with inf.
        """
        max_bd = 16  # 4-bit = 15 boundaries max, pad to 16 for alignment
        all_bd = torch.full((self.n_levels, max_bd), float('inf'), dtype=torch.float32)
        for lv in range(self.n_levels):
            bits = self._get_bits(lv)
            bd = get_angular_boundaries(bits, lv)
            n = min(bd.shape[0], max_bd)
            all_bd[lv, :n] = bd[:n]
        return all_bd

    def get_all_centroids(self) -> torch.Tensor:
        """
        Return a flat tensor of all level centroids for Triton kernels.
        Shape: (n_levels, max_centroids) padded with 0.
        """
        max_ct = 16  # 4-bit = 16 centroids max
        all_ct = torch.zeros((self.n_levels, max_ct), dtype=torch.float32)
        for lv in range(self.n_levels):
            bits = self._get_bits(lv)
            cb = get_angular_codebook(bits, lv)
            n = min(cb.shape[0], max_ct)
            all_ct[lv, :n] = cb[:n]
        return all_ct
