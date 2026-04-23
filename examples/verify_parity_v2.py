import torch
import math
import sys
import os

# Ensure we can import tq_impl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoConfig, AutoModelForCausalLM
from tq_impl import TurboQuantCache, patch_model_for_turboquant

def verify_parity(model_id="Qwen/Qwen2.5-0.5B-Instruct"):
    print(f"--- Verifying Parity for {model_id} ---")
    device = "cuda"
    dtype = torch.float16
    
    # 1. Setup Cache
    cache = TurboQuantCache(bits_key=4.0, outliers=True, dtype=dtype)
    
    # 2. Mock Data
    # B, H_q, H_kv, T, D
    B, H_q, H_kv, T = 1, 14, 2, 128
    config = AutoConfig.from_pretrained(model_id)
    D = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    
    # Random KV in original space
    k = torch.randn(B, H_kv, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H_kv, T, D, device=device, dtype=dtype)
    q = torch.randn(B, H_q, 1, D, device=device, dtype=dtype)
    
    layer_idx = 0
    
    # 3. Compress KV
    print(f"Compressing KV (D={D})...")
    # Simulate prefill
    cache.update(k, v, layer_idx)
    
    # 4. Compute Python Reconstructed Score
    print("Computing Python reference score...")
    k_rec, v_rec = cache.update(torch.empty((B, H_kv, 0, D), device=device, dtype=dtype), 
                                torch.empty((B, H_kv, 0, D), device=device, dtype=dtype), 
                                layer_idx)
    
    # GQA Repeat for Python
    k_rec_rep = k_rec.repeat_interleave(H_q // H_kv, dim=1)
    # k_rec_rep shape: [B, H_q, T, D]
    # score = q * k^T
    # q is [B, H_q, 1, D]
    ref_scores = torch.matmul(q, k_rec_rep.transpose(-1, -2)) # [B, H_q, 1, T]
    
    # 5. Compute Triton Fused Score
    print("Computing Triton fused score...")
    fused_scores = cache.fused_scores(q, layer_idx) # [B, H_q, 1, T]
    
    # 6. Compare
    diff = (ref_scores - fused_scores).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nResults (D={D}):")
    print(f"  Max Diff:  {max_diff:.8f}")
    print(f"  Mean Diff: {mean_diff:.8f}")
    
    if max_diff < 1e-3:
        print("✅ SUCCESS: Triton matches Python (Elite V3 Parity OK)")
    else:
        print("❌ FAILURE: Numerical divergence detected!")
        # Debug indices
        if max_diff > 0.1:
            idx = torch.argmax(diff)
            print(f"  Large error at flattened index {idx}")

if __name__ == "__main__":
    model = "Qwen/Qwen2.5-7B-Instruct" if len(sys.argv) < 2 else sys.argv[1]
    verify_parity(model)
