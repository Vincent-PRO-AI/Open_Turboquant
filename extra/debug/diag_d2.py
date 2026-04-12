import torch
import math
from tq_impl.triton_polar import triton_polar_encode, triton_polar_decode, is_triton_available
from tq_impl.polar import recursive_polar_transform, recursive_polar_inverse
from tq_impl.polar_quant import PolarAngleQuantizer

def diagnose_d2():
    device = "cuda"
    D = 2
    x = torch.randn(1, 1, 1, D, device=device, dtype=torch.float32)
    pq = PolarAngleQuantizer(d=D) # L=1
    
    print(f"--- D={D} ENCODER CHECK ---")
    rf_py, angs_py = recursive_polar_transform(x)
    rf_tr, pa_tr = triton_polar_encode(x, pq.get_all_boundaries(), D)
    
    print(f"Radius Diff: {(rf_py - rf_tr).abs().max().item():.2e}")
    print(f"Angle Index Diff: {(pq.quantize_all(angs_py)[0].to(torch.int32) - pa_tr[0].to(torch.int32)).abs().max().item()}")

    print(f"\n--- D={D} DECODER CHECK ---")
    x_rec_py = recursive_polar_inverse(rf_py, pq.dequantize_all(pq.quantize_all(angs_py)))
    x_rec_tr = triton_polar_decode(rf_tr, pa_tr, pq.get_all_centroids(), D)
    
    cos_sim = torch.nn.functional.cosine_similarity(x_rec_py.view(-1), x_rec_tr.view(-1), dim=0)
    print(f"Inverse CosSim: {cos_sim.item():.6f}")
    print(f"X[0]: PY={x_rec_py[0,0,0,0]:.4f}, TR={x_rec_tr[0,0,0,0]:.4f}")
    print(f"X[1]: PY={x_rec_py[0,0,0,1]:.4f}, TR={x_rec_tr[0,0,0,1]:.4f}")

if __name__ == "__main__":
    diagnose_d2()
