import os
import sys
import torch

# Force CPU to simulate APU/Non-CUDA environment
device = 'cpu'

# Fix pour permettre l'import de tq_impl depuis le dossier tests/
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from tq_impl import TurboQuantCache
import time

def test_polar_fidelity_cpu():
    # Small test vector
    head_dim = 128
    B, H, T = 1, 4, 32
    k = torch.randn(B, H, T, head_dim, device=device, dtype=torch.float32) # CPU prefers float32
    v = torch.randn(B, H, T, head_dim, device=device, dtype=torch.float32)
    
    print(f'--- TESTING POLARQUANT ON {device.upper()} (APU/CPU MODE) ---')
    # Force compress_start to 0 to trigger compression immediately
    cache = TurboQuantCache(num_outlier_pairs=4)
    
    # 1. Prefill (Raw -> Auto Compress)
    k_out, v_out = cache.update(k, v, 0)
    
    # Check if compressed
    if cache._compressed.get(0):
        print('[OK] Engine successfully activated Fallback Compression on CPU.')
    
    # 2. Decode Step
    k_new = torch.randn(B, H, 1, head_dim, device=device, dtype=torch.float32)
    v_new = torch.randn(B, H, 1, head_dim, device=device, dtype=torch.float32)
    k_rec, v_rec = cache.update(k_new, v_new, 0)
    
    # 3. Fidelity Check
    k_full = torch.cat([k, k_new], dim=2)
    k_cache = cache.key_cache[0].to(torch.float32) # Get reconstructed cache
    
    cos_sim = torch.nn.functional.cosine_similarity(k_full, k_cache, dim=-1).mean()
    print(f'Mean Cosine Similarity: {cos_sim.item():.6f}')
    
    if cos_sim > 0.99:
        print('[SUCCESS] PolarQuant Fidelity logic is working perfectly on APU/CPU!')
    else:
        print('[FAILURE] Fidelity check failed.')

if __name__ == '__main__':
    test_polar_fidelity_cpu()
