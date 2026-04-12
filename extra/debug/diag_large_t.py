
import torch
import torch.nn.functional as F
from tq_impl.triton_polar import triton_polar_encode, triton_polar_decode
from tq_impl.polar_quant import PolarAngleQuantizer

def diag_large_t():
    # Use real benchmark sizes
    B, H, T, D = 1, 32, 2048, 128
    print(f"Testing with B={B}, H={H}, T={T}, D={D}")
    
    device = 'cuda'
    dtype = torch.float16
    x = torch.randn(B, H, T, D, device=device, dtype=dtype)
    
    pq = PolarAngleQuantizer(d=D)
    bd = pq.get_all_boundaries().to(device)
    ct = pq.get_all_centroids().to(device)
    
    print("Running Encode...")
    rf, pa = triton_polar_encode(x, bd, D)
    if torch.isnan(rf).any():
        print("!! NaNs in Radii")
    
    print("Running Decode...")
    x_rec = triton_polar_decode(rf, pa, ct, D)
    if torch.isnan(x_rec).any():
        print("!! NaNs in Reconstruction")
    
    cos = F.cosine_similarity(x.float(), x_rec.float(), dim=-1).mean()
    print(f"CosSim: {cos.item():.6f}")

if __name__ == "__main__":
    diag_large_t()
