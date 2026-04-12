import torch
import math
from tq_impl.triton_polar import triton_polar_encode, triton_polar_decode, is_triton_available
from tq_impl.polar import recursive_polar_transform, recursive_polar_inverse
from tq_impl.polar_quant import PolarAngleQuantizer

def diagnose_d32():
    device = "cuda"
    D = 32
    # L=5. Levels: 0, 1, 2, 3 (4-bit), 4 (2-bit).
    x = torch.randn(1, 1, 1, D, device=device, dtype=torch.float32)
    pq = PolarAngleQuantizer(d=D)
    
    print(f"--- D={D} ENCODER CHECK ---")
    rf_py, angs_py = recursive_polar_transform(x)
    rf_tr, pa_tr = triton_polar_encode(x, pq.get_all_boundaries(), D)
    
    print(f"Radius Diff: {(rf_py - rf_tr).abs().max().item():.2e}")
    for i in range(len(pa_tr)):
        diff = (pa_tr[i].view(-1).to(torch.int32) - pq.pack_all(pq.quantize_all(angs_py))[i].view(-1).to(torch.int32)).abs().max().item()
        print(f"Level {i} ({'4-bit' if i<=3 else '2-bit'}) Angle Chunk Diff: {diff}")

    print(f"\n--- D={D} DECODER CHECK ---")
    idx_py = pq.quantize_all(angs_py)
    pa_py = pq.pack_all(idx_py)
    x_rec_py = recursive_polar_inverse(rf_py, pq.dequantize_all(pq.unpack_all(pa_py)))
    x_rec_tr = triton_polar_decode(rf_tr, pa_tr, pq.get_all_centroids(), D)
    
    cos_sim = torch.nn.functional.cosine_similarity(x_rec_py.view(-1), x_rec_tr.view(-1), dim=0)
    print(f"Inverse CosSim: {cos_sim.item():.6f}")

if __name__ == "__main__":
    diagnose_d32()
