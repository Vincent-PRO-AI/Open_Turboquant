import torch
import math
from typing import Tuple, List

def cartesian_to_polar(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert (x, y) to (r, phi). phi is in [0, 2*pi]."""
    r = torch.sqrt(x**2 + y**2 + 1e-12)
    phi = torch.atan2(y, x)
    # Ensure phi in [0, 2*pi]
    phi = torch.where(phi < 0, phi + 2 * math.pi, phi)
    return r.to(x.dtype), phi.to(x.dtype)

def polar_to_cartesian(r: torch.Tensor, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert (r, phi) to (x, y)."""
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    return x.to(r.dtype), y.to(r.dtype)

def recursive_polar_transform(x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Applies the recursive polar transformation.
    x shape: (..., d) where d is power of 2.
    Returns:
        final_radius: (..., 1)
        angles: List of tensors, each of shape (..., d/2^(level+1))
    """
    orig_shape = x.shape
    d = x.shape[-1]
    n_levels = int(math.log2(d))
    current_radii = x
    all_angles = []

    for level in range(n_levels):
        # M = d / 2^(level+1) pairs
        # Reshape to (..., M, 2)
        m = current_radii.shape[-1] // 2
        pairs = current_radii.reshape(*current_radii.shape[:-1], m, 2)
        r, phi = cartesian_to_polar(pairs[..., 0], pairs[..., 1])
        all_angles.append(phi)
        current_radii = r

    return current_radii, all_angles

def recursive_polar_inverse(final_radius: torch.Tensor, angles: List[torch.Tensor]) -> torch.Tensor:
    """
    Reconstructs the original vector from final radius and angle tree.
    """
    current_radii = final_radius
    # Traverse angles in reverse order
    for level_i, phi in enumerate(reversed(angles)):
        # current_radii is (..., M), phi is (..., M)
        if current_radii.shape != phi.shape:
            raise RuntimeError(
                f"[polar_inverse] Shape mismatch at reverse level {level_i}: "
                f"radii={list(current_radii.shape)} vs phi={list(phi.shape)}"
            )
        x, y = polar_to_cartesian(current_radii, phi)
        # Combine back into (..., M*2)
        current_radii = torch.stack([x, y], dim=-1).reshape(*x.shape[:-1], -1)
    
    return current_radii

# Simple test
if __name__ == "__main__":
    d = 128
    x = torch.randn(2, 8, 32, d) # (B, H, T, d)
    r, angles = recursive_polar_transform(x)
    x_rec = recursive_po