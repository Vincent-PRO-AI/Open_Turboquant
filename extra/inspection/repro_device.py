import torch
import math
from tq_impl.cache import TurboQuantCache

def test_device_issue():
    device = "cuda:0"
    B, H, T, D = 1, 8, 128, 128
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    
    cache = TurboQuantCache(bits=4.0, dtype=torch.float16)
    print(f"Update prefill...")
    k_rec, v_rec = cache.update(k, v, 0)
    print(f"Prefill done. Keys device: {k_rec.device}")
    
    # Test decode
    k_new = torch.randn(B, H, 1, D, device=device, dtype=torch.float16)
    v_new = torch.randn(B, H, 1, D, device=device, dtype=torch.float16)
    print(f"Update decode (T=1)...")
    k_rec2, v_rec2 = cache.update(k_new, v_new, 0)
    print(f"Decode done. Keys device: {k_rec2.device}")

if __name__ == "__main__":
    test_device_issue()
