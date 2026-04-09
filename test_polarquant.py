import torch
from tq_impl import TurboQuantCache
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def test_polar_fidelity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Small test vector
    head_dim = 128
    B, H, T = 1, 4, 32
    k = torch.randn(B, H, T, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(B, H, T, head_dim, device=device, dtype=torch.float16)
    
    print("Testing PolarQuant Fidelity...")
    cache = TurboQuantCache(num_outlier_pairs=4)
    
    # 1. Prefill (Raw)
    k_out, v_out = cache.update(k, v, 0)
    print(f"Prefill diff: {(k - k_out).abs().max().item():.2e}")
    
    # 2. Force Compression
    cache._compress_layer(0)
    print("Layer 0 compressed to Polar format.")
    
    # 3. Decode Step
    k_new = torch.randn(B, H, 1, head_dim, device=device, dtype=torch.float16)
    v_new = torch.randn(B, H, 1, head_dim, device=device, dtype=torch.float16)
    k_rec, v_rec = cache.update(k_new, v_new, 0)
    
    # 4. Check Cosine Similarity of the entire cache
    k_full = torch.cat([k, k_new], dim=2)
    # Reconstruct from cache
    k_cache = cache.key_cache[0]
    
    cos_sim = torch.nn.functional.cosine_similarity(k_full, k_cache, dim=-1).mean()
    print(f"Mean Cosine Similarity: {cos_sim.item():.6f}")
    
    if cos_sim > 0.99:
        print("✅ Fidelity check passed!")
    else:
        print("❌ Fidelity check failed!")

if __name__ == "__main__":
    test_polar_fidelity()
