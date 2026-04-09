import torch
import math
import numpy as np
from tq_impl.polar import recursive_polar_transform, recursive_polar_inverse
from tq_impl.triton_polar import triton_polar_encode, triton_polar_decode, is_triton_available
from tq_impl.polar_quant import PolarAngleQuantizer

def test_parity():
    if not is_triton_available():
        print("Triton not available")
        return

    B, H, T, D = 1, 8, 1, 128
    x = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
    
    pq = PolarAngleQuantizer(d=D)
    boundaries = pq.get_all_boundaries()
    centroids = pq.get_all_centroids()

    # Triton path
    r_tr, p_tr = triton_polar_encode(x, boundaries, D)
    x_rec_tr = triton_polar_decode(r_tr, p_tr, centroids, D)

    # PyTorch path
    r_py, ang_py = recursive_polar_transform(x)
    idx_py = pq.quantize_all(ang_py)
    p_py = pq.pack_all(idx_py)
    
    # Dequantize for PyTorch
    unpacked_py = pq.unpack_all(p_py)
    rec_angs_py = pq.dequantize_all(unpacked_py)
    x_rec_py = recursive_polar_inverse(r_py, rec_angs_py)

    print(f"Stats for {D} dimensions:")
    print(f"X range: [{x.min().item():.3f}, {x.max().item():.3f}]")
    
    cos_tr = torch.nn.functional.cosine_similarity(x.flatten(), x_rec_tr.flatten(), dim=0).item()
    cos_py = torch.nn.functional.cosine_similarity(x.flatten(), x_rec_py.flatten(), dim=0).item()
    cos_cross = torch.nn.functional.cosine_similarity(x_rec_tr.flatten(), x_rec_py.flatten(), dim=0).item()
    
    print(f"Triton CosSim: {cos_tr:.6f}")
    print(f"PyTorch CosSim: {cos_py:.6f}")
    print(f"Cross-Parity CosSim: {cos_cross:.6f}")
    
    # Inspection
    print(f"\nLevel 0 Radius (first 4):")
    # In PyTorch, radii of level 0 are the output of the first recursive call
    # We can't easily get it without patching polar.py, so we'll check final radii instead
    print(f"Final Radius Triton: {r_tr[0,0,0,0].item():.6f}")
    print(f"Final Radius PyTorch: {r_py[0,0,0,0].item():.6f}")

    print("\nLevel 0 Packed (first 8 bytes):")
    print(f"Triton : {p_tr[0][0,0,0,:8].tolist()}")
    print(f"PyTorch: {p_py[0][0,0,0,:8].tolist()}")

    print("\nFirst 8 elements (X):")
    print(f"Orig   : {x[0,0,0,:8].tolist()}")
    print(f"Triton : {x_rec_tr[0,0,0,:8].tolist()}")
    print(f"PyTorch: {x_rec_py[0,0,0,:8].tolist()}")

    print("\nElements 64-71 (X):")
    print(f"Triton : {x_rec_tr[0,0,0,64:72].tolist()}")
    print(f"PyTorch: {x_rec_py[0,0,0,64:72].tolist()}")

    # Compare raw angles
    r_diff = (r_tr - r_py).abs().max().item()
    print(f"\nMax Radii Diff: {r_diff:.6e}")
    
    # Check centroids and boundaries
    cb_tr = centroids
    cb_py = pq.get_all_centroids()
    for i in range(len(cb_tr)):
        c_diff = (cb_tr[i].cpu() - cb_py[i].cpu()).abs().max().item()
        if c_diff > 1e-5:
            print(f"Centroids mismatch at level {i}: {c_diff}")

if __name__ == "__main__":
    test_parity()
