import torch
import math
from tq_impl.triton_polar import triton_polar_encode, triton_polar_decode, is_triton_available
from tq_impl.polar import recursive_polar_transform, recursive_polar_inverse
from tq_impl.polar_quant import PolarAngleQuantizer

def diagnose_all_values():
    device = "cuda"
    D = 128
    x = torch.ones(1, 1, 1, D, device=device, dtype=torch.float32)
    pq = PolarAngleQuantizer(d=D)
    
    rf_py, angs_py = recursive_polar_transform(x)
    rf_tr, pa_tr = triton_polar_encode(x, pq.get_all_boundaries(), D)
    
    x_rec_py = recursive_polar_inverse(rf_py, pq.dequantize_all(pq.unpack_all(pa_tr)))
    x_rec_tr = triton_polar_decode(rf_tr, pa_tr, pq.get_all_centroids(), D)
    
    diff = (x_rec_py - x_rec_tr).abs().view(-1)
    print(f"Max Diff: {diff.max().item():.2e}")
    print(f"Indices with large diff: {torch.where(diff > 1e-4)[0].tolist()[:10]}")
    
    cos_sim = torch.nn.functional.cosine_similarity(x_rec_py.view(-1), x_rec_tr.view(-1), dim=0)
    print(f"Inverse CosSim: {cos_sim.item():.6f}")

if __name__ == "__main__":
    diagnose_all_values()
