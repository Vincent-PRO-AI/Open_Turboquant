import torch
import gc
import json
import time
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from tq_impl import TurboQuantCache, patch_model_for_turboquant

MODEL_ID = "google/gemma-4-26B-A4B-it"

def get_vram_usage():
    # Sum across all GPUs
    total = 0
    for i in range(torch.cuda.device_count()):
        total += torch.cuda.max_memory_allocated(i)
    return total / (1024**3)

def stress_test(mode="baseline"):
    print(f"\n🚀 Starting MoE Stress Test [Mode: {mode}]")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load model across all available GPUs
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    results = []
    # Test levels from 10k to 1.5M tokens
    test_levels = [10000, 50000, 100000, 200000, 300000, 500000, 750000, 1000000, 1250000, 1500000]
    
    last_success = 0
    
    try:
        for ctx_len in test_levels:
            print(f"Testing context length: {ctx_len} tokens...")
            torch.cuda.reset_peak_memory_stats()
            
            if mode == "turboquant":
                # Create TurboQuant cache
                cache = TurboQuantCache(bits=4.0, dtype=model.dtype, max_seq_len=ctx_len)
                # No need to patch every time, but ensure the cache object is brand new
            else:
                # Mock a standard cache by allocating the tensors
                # We don't use DynamicCache because it grows. We want to measure the peak of a FIXED size for baseline too.
                # A standard FP16 KV cache for this model:
                # Num layers: 35 (Gemma-4)
                # Num heads: 8 (GQA)
                # Head dim: 256
                # Total: layers * 2 (K,V) * heads * seq * dim * 2 bytes
                # Num layers: Detection for Gemma-4 / Others
                layers = getattr(model.config, 'num_hidden_layers', getattr(model.config, 'num_layers', 35))
                heads = getattr(model.config, 'num_key_value_heads', getattr(model.config, 'num_attention_heads', 8))
                dim = getattr(model.config, 'head_dim', 256)
                
                # Allocation simulation (the most accurate way to find OOM)
                k_cache = torch.zeros((1, heads, ctx_len, dim), dtype=torch.bfloat16, device="cuda")
                v_cache = torch.zeros((1, heads, ctx_len, dim), dtype=torch.bfloat16, device="cuda")
                # Total layers (this is what triggers OOM)
                dummy_list = [torch.zeros_like(k_cache) for _ in range(layers * 2)]
            
            vram = get_vram_usage()
            print(f"  VRAM Usage: {vram:.2f} GB")
            results.append({"ctx": ctx_len, "vram": vram})
            last_success = ctx_len
            
            # Cleanup for next iteration
            if mode == "turboquant":
                del cache
            else:
                del dummy_list
            gc.collect()
            torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        print(f"❌ OOM reached at {ctx_len} tokens!")
        results.append({"ctx": ctx_len, "status": "OOM"})

    # Complete cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return results, last_success

if __name__ == "__main__":
    final_report = {}
    
    # Run Baseline
    baseline_data, b_max = stress_test(mode="baseline")
    final_report["baseline"] = baseline_data
    
    # Run TurboQuant
    tq_data, tq_max = stress_test(mode="turboquant")
    final_report["turboquant"] = tq_data
    
    with open("moe_bench_results.json", "w") as f:
        json.dump(final_report, f, indent=2)
    
    print("\n✅ Stress test complete. Results saved to moe_bench_results.json")
    print(f"Baseline Max: {b_max} tokens")
    print(f"TurboQuant Max: {tq_max} tokens")
