import torch
import math
from tq_impl.triton_polar import triton_polar_encode, triton_polar_decode, is_triton_available
from tq_impl.polar import recursive_polar_transform, recursive_polar_inverse
from tq_impl.polar_quant import PolarAngleQuantizer

def diagnose():
    device = "cuda"
    D = 128
    x = torch.randn(1, 1, 1, D, device=device, dtype=torch.float32)
    pq = PolarAngleQuantizer(d=D)
    
    print("--- ENCODER CHECK ---")
    # PyTorch
    rf_py, angs_py = recursive_polar_transform(x)
    idx_py = pq.quantize_all(angs_py)
    pa_py = pq.pack_all(idx_py)
    
    # Triton
    rf_tr, pa_tr = triton_polar_encode(x, pq.get_all_boundaries(), D)
    
    print(f"Radius Diff: {(rf_py - rf_tr).abs().max().item():.2e}")
    for i in range(len(pa_py)):
        print(f"Level {i} Angle Diff (Packed Bits): {(pa_py[i].to(torch.int32) - pa_tr[i].to(torch.int32)).abs().max().item()}")

    print("\n--- DECODER CHECK ---")
    # PyTorch
    x_rec_py = recursive_polar_inverse(rf_py, pq.dequantize_all(idx_py))
    # Triton
    x_rec_tr = triton_polar_decode(rf_tr, pa_tr, pq.get_all_centroids(), D)
    
    cos_sim = torch.nn.functional.cosine_similarity(x_rec_py.view(-1), x_rec_tr.view(-1), dim=0)
    print(f"Inverse CosSim (TR vs PY): {cos_sim.item():.6f}")
    
    print(f"Final Value Diff (max): {(x_rec_py - x_rec_tr).abs().max().item():.2e}")

if __name__ == "__main__":
    if is_triton_available():
        diagnose()
    else:
        print("Triton not available.")
