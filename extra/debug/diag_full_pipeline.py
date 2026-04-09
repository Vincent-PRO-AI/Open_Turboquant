
import torch
import torch.nn.functional as F
from tq_impl.cache import TurboQuantCache
import math

def diag_full_pipeline():
    print("=== TurboQuant v2 Full Pipeline Diagnostic ===")
    B, H, D = 1, 32, 128
    T_prefill = 512
    T_decode = 10
    
    device = 'cuda'
    dtype = torch.float16
    
    # 1. Initialize Cache
    cache = TurboQuantCache(bits=4.0, dtype=dtype)
    
    # 2. Simulate Prefill
    print(f"Phase 1: Prefill (T={T_prefill})")
    k_pre = torch.randn(B, H, T_prefill, D, device=device, dtype=dtype)
    v_pre = torch.randn(B, H, T_prefill, D, device=device, dtype=dtype)
    
    # Prefill usually goes through standard update or update_compressed
    # In run_benchmark_v3, we use model.generate which calls update().
    # But for quality checks it might call update_compressed.
    try:
        cache.update_compressed(k_pre, v_pre, layer_idx=0)
        print("  Prefill update_compressed successful.")
    except Exception as e:
        print(f"  !! Prefill Error: {e}")
        return

    # 3. Simulate Decode
    print(f"Phase 2: Decode (T={T_decode} steps)")
    for t in range(T_decode):
        k_new = torch.randn(B, H, 1, D, device=device, dtype=dtype)
        v_new = torch.randn(B, H, 1, D, device=device, dtype=dtype)
        q_new = torch.randn(B, H, 1, D, device=device, dtype=dtype)
        
        # update_compressed (what fused_decode does)
        cache.update_compressed(k_new, v_new, layer_idx=0)
        
        # fused_scores
        scores = cache.fused_scores(q_new, layer_idx=0)
        
        if torch.isnan(scores).any():
            print(f"  !! Step {t}: NaNs detected in scores!")
            # Find which branch has NaNs
            # (Repeating the math to isolate)
            sk = cache._sketch_matrices[0]
            k_rec_sk = cache._reconstruct_keys_sketched(0)
            q_sk = torch.matmul(q_new, sk)
            scores_mse = torch.matmul(q_sk, k_rec_sk.transpose(-1, -2))
            if torch.isnan(scores_mse).any(): print("    NaN in MSE branch")
            
            proj = cache._qjl_projections[0]
            q_p = torch.matmul(q_new, proj)
            q_signs = torch.sign(q_p)
            k_signs = cache.get_seq_length(0) # simplified check
            # ...
            break
        
        if t % 5 == 0:
            print(f"  Step {t}: Scores Max={scores.max().item():.4f}, Min={scores.min().item():.4f}")

    print("\nState Summary:")
    print(f"  Cache Length: {cache.get_seq_length(0)}")
    print(f"  Final Radii Max: {cache._final_radii[0].max().item():.4f}")
    
    # Final check on reconstruction quality
    k_rec = cache.key_cache[0]
    cos_sim = F.cosine_similarity(k_pre.float(), k_rec[:,:,:T_prefill,:].float(), dim=-1).mean()
    print(f"  Reconstruction CosSim: {cos_sim.item():.6f}")

if __name__ == "__main__":
    diag_full_pipeline()
