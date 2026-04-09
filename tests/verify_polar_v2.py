import torch
import math
from tq_impl.polar import recursive_polar_transform, recursive_polar_inverse
from tq_impl.polar_quant import PolarAngleQuantizer

def verify_v2():
    d = 128
    B, KVH, T = 1, 4, 32
    head_dim = d
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Generate random keys
    k = torch.randn(B, KVH, T, head_dim, device=device)
    k = k / k.norm(dim=-1, keepdim=True) # unit sphere for simplicity
    
    # 2. Transform to Polar
    r_final, angles = recursive_polar_transform(k)
    
    # 3. Quantize with Hierarchy (4-bit L0, 2-bit others)
    pq = PolarAngleQuantizer(d=head_dim)
    indices = pq.quantize_all(angles)
    
    # 4. Pack and Unpack
    packed = pq.pack_all(indices)
    
    # Print shapes to verify bit-packing
    print(f"Original head_dim: {head_dim}")
    for i, p in enumerate(packed):
        bits = 4 if i == 0 else 2
        pack_factor = 8 // bits
        print(f"Level {i}: packed shape {p.shape}, bits {bits}, factor {pack_factor}")

    unpacked = pq.unpack_all(packed)
    
    # 5. Reconstruct
    rec_angles = pq.dequantize_all(unpacked)
    k_rec = recursive_polar_inverse(r_final, rec_angles)
    
    # 6. Metrics
    cos = torch.nn.functional.cosine_similarity(k, k_rec, dim=-1).mean().item()
    mse = ((k - k_rec)**2).mean().item()
    
    print(f"\nPolarQuant v2 Metrics:")
    print(f"Cosine Similarity: {cos:.6f}")
    print(f"MSE: {mse:.6e}")
    
    assert cos > 0.95, f"Cosine similarity too low: {cos}"
    print("\nVerification PASSED!")

if __name__ == "__main__":
    verify_v2()
