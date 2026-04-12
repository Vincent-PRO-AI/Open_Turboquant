import torch
import math
from tq_impl.cache import TurboQuantCache

def test_polar_fidelity():
    print("Testing PolarQuant Fidelity (Identity Sketch)...")
    B, H, T, D = 1, 8, 128, 128
    device = "cuda"
    
    # Correct Init
    cache = TurboQuantCache(num_outlier_pairs=0) # No outliers
    
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    
    # 1. First Update to trigger resource allocation
    cache.update(k, v, 0)
    
    # 2. Forced Identity Sketch on Layer 0
    if 0 in cache._sketch_matrices:
        cache._sketch_matrices[0].zero_()
        cache._sketch_matrices[0].fill_diagonal_(1.0)
        print("Forced Identity Sketch on Layer 0.")
    
    # 3. Second Update with Identity Sketch (Pre-filling)
    # We need to clear the previous cache state for Layer 0 if we want a clean identity test
    cache._values.clear()
    cache._raw_keys.clear()
    cache._final_radii.clear()
    cache._packed_angles.clear()
    cache._compressed = {}
    
    cache.update(k, v, 0)
    
    # In TurboQuantCache, the key_cache property reconstructs based on _final_radii or _raw_keys.
    # If T > 1, it stores in _raw_keys. To test the compression, we need to call with T=1 OR 
    # force compression.
    
    # Force compression of the raw keys
    cache._compress_layer(0)
    
    k_rec = cache.key_cache[0]
    
    cos_sim = torch.nn.functional.cosine_similarity(k.view(-1).to(torch.float32), k_rec.view(-1).to(torch.float32), dim=0)
    print(f"Mean Cosine Similarity: {cos_sim.item():.6f}")
    
    if cos_sim > 0.99:
        print("✅ Fidelity check passed!")
    else:
        print("❌ Fidelity check failed!")

if __name__ == "__main__":
    test_polar_fidelity()
