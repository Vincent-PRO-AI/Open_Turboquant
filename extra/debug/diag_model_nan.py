
import torch
import torch.nn.functional as F
from tq_impl.triton_polar import triton_polar_encode, triton_polar_decode
from tq_impl.polar_quant import PolarAngleQuantizer

def diag_model_nan():
    # Use real model-like ranges
    B, H, T, D = 1, 32, 1, 128
    x = torch.randn(B, H, T, D, device='cuda', dtype=torch.float16) * 2.0
    
    pq = PolarAngleQuantizer(d=D)
    bd = pq.get_all_boundaries().cuda()
    ct = pq.get_all_centroids().cuda()
    
    print("Testing Encode...")
    try:
        rf, pa = triton_polar_encode(x, bd, D)
        print(f"  Radii Mean: {rf.mean().item():.4f}, Max: {rf.max().item():.4f}")
        if torch.isnan(rf).any():
            print("  !! ERROR: Nan in Radii")
    except Exception as e:
        print(f"  !! Encode Error: {e}")

    print("Testing Decode...")
    try:
        x_rec = triton_polar_decode(rf, pa, ct, D)
        print(f"  Rec Mean: {x_rec.mean().item():.4f}, Max: {x_rec.max().item():.4f}")
        if torch.isnan(x_rec).any():
            print("  !! ERROR: Nan in Reconstructed")
        
        cos = F.cosine_similarity(x.float(), x_rec.float(), dim=-1).mean()
        print(f"  CosSim: {cos.item():.6f}")
    except Exception as e:
        print(f"  !! Decode Error: {e}")

if __name__ == "__main__":
    diag_model_nan()
