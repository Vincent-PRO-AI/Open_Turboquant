import torch
import math
from tq_impl.triton_polar import triton_polar_encode, triton_polar_decode, is_triton_available
from tq_impl.polar import recursive_polar_transform, recursive_polar_inverse
from tq_impl.polar_quant import PolarAngleQuantizer

def diagnose_d128_ones():
    device = "cuda"
    D = 128
    # Test with all-ones to check if magnitudes are preserved
    x = torch.ones(1, 1, 1, D, device=device, dtype=torch.float32)
    pq = PolarAngleQuantizer(d=D)
    
    print(f"--- D={D} ONES CHECK ---")
    rf_py, angs_py = recursive_polar_transform(x)
    rf_tr, pa_tr = triton_polar_encode(x, pq.get_all_boundaries(), D)
    
    print(f"Radius Diff: {(rf_py - rf_tr).abs().max().item():.2e}")
    
    x_rec_py = recursive_polar_inverse(rf_py, pq.dequantize_all(pq.unpack_all(pa_tr)))
    x_rec_tr = triton_polar_decode(rf_tr, pa_tr, pq.get_all_centroids(), D)
    
    cos_sim = torch.nn.functional.cosine_similarity(x_rec_py.view(-1), x_rec_tr.view(-1), dim=0)
    print(f"Inverse CosSim (Ones): {cos_sim.item():.6f}")

if __name__ == "__main__":
    diagnose_d128_ones()
