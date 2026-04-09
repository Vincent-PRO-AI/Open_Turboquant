import os
import sys
import torch

# Fix pour permettre l'import de tq_impl depuis le dossier tests/
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

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
    
    # 2. Status Check (Compression is automatic in v1.0)
    if cache._compressed.get(0):
        print("[OK] Layer 0 automatically compressed to Polar format.")
    
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
        print("[SUCCESS] Fidelity check passed!")
    else:
        print("[FAILURE] Fidelity check failed!")

if __name__ == "__main__":
    test_polar_fidelity()
