
import torch
import math
import numpy as np
from tq_impl.triton_polar import triton_polar_encode
from tq_impl.polar import recursive_polar_transform
from tq_impl.polar_quant import PolarAngleQuantizer
from tq_impl.codebook import get_boundaries

def diag_levels():
    D = 128
    L = 7
    x = torch.randn(1, 1, 1, D, device='cuda', dtype=torch.float32)
    
    # Get boundaries
    pq = PolarAngleQuantizer(d=D)
    boundaries = pq.get_all_boundaries().cuda()
    
    # Reference
    rf_py, angs_py = recursive_polar_transform(x)
    idx_py = pq.quantize_all(angs_py)
    
    # Triton
    rf_tr, packed_tr = triton_polar_encode(x, boundaries, D)
    
    print(f"D={D} Final Radius Py: {rf_py.squeeze().item():.6f}")
    print(f"D={D} Final Radius Tr: {rf_tr.squeeze().item():.6f}")
    
    for lv in range(L):
        bits = 4 if lv <= 3 else 2
        p = packed_tr[lv].cpu()
        idx_tr = []
        if bits == 4:
            for b in p.flatten():
                idx_tr.append(b & 0x0F)
                idx_tr.append((b >> 4) & 0x0F)
        else:
            for b in p.flatten():
                idx_tr.append(b & 0x03)
                idx_tr.append((b >> 2) & 0x03)
                idx_tr.append((b >> 4) & 0x03)
                idx_tr.append((b >> 6) & 0x03)
        
        py_vals = idx_py[lv].flatten().tolist()
        tr_vals = idx_tr[:len(py_vals)]
        matches = (np.array(py_vals) == np.array(tr_vals)).all()
        print(f"Level {lv} ({bits}-bit) Matches: {matches}")
        if not matches:
            print(f"  Py: {py_vals}")
            print(f"  Tr: {tr_vals}")

if __name__ == "__main__":
    diag_levels()
