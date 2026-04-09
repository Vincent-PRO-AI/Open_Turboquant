#!/usr/bin/env python3
"""
comprehensive_benchmark.py — The ultimate PolarQuant vs Baseline Benchmarking Tool
===================================================================================

Measures:
- Prefill Latency (TTFT)
- Decode Throughput (TPS)
- VRAM Footprint & Key Compression Ratio
- Numerical Fidelity (CosSim, Top-1)
- Qualitative Generation Samples
"""

import gc, sys, time, math, os, json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(__file__))
from tq_impl import (
    TurboQuantCache,
    patch_model_for_turboquant, unpatch_model_for_turboquant,
    compression_ratio
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

MODELS = ["Qwen/Qwen2.5-7B-Instruct", "google/gemma-4-E2B-it"]
MODES = ["baseline", "tq4b", "tq3b"]
CONTEXT_SIZES = [1024, 4096] # Stress test points
GEN_TOKENS = 64

results = {}

def get_vram():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated(0) / 1024**3, torch.cuda.max_memory_allocated(0) / 1024**3

def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()

def measure_step(model, tokenizer, ids, bits=None, label="baseline"):
    clear_vram()
    v_start, _ = get_vram()
    
    cache = None
    if bits:
        cache = TurboQuantCache(bits=float(bits), dtype=model.dtype)
        patch_model_for_turboquant(model, cache)
    
    try:
        # 1. PREFILL
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model(ids, past_key_values=cache, use_cache=True)
            prefill_logits = outputs.logits[:, -1, :]
        torch.cuda.synchronize()
        t_pre = (time.perf_counter() - t0) * 1000 # ms
        
        # 2. DECODE
        t1 = time.perf_counter()
        with torch.inference_mode():
            gen_out = model.generate(
                ids, 
                past_key_values=cache, 
                max_new_tokens=GEN_TOKENS, 
                do_sample=False, 
                use_cache=True
            )
        torch.cuda.synchronize()
        t_dec = (time.perf_counter() - t1) # seconds
        
        v_end, v_peak = get_vram()
        kv_usage = v_end - v_start
        
        # 3. SAMPLE
        sample_text = tokenizer.decode(gen_out[0][-GEN_TOKENS:], skip_special_tokens=True)
        
        return {
            "prefill_ms": t_pre,
            "tps": GEN_TOKENS / t_dec,
            "vram_peak": v_peak,
            "kv_vram": kv_usage,
            "sample": sample_text,
            "logits": prefill_logits
        }
    except torch.cuda.OutOfMemoryError:
        print(f"      [!] OOM for {label}")
        return None
    finally:
        if bits: unpatch_model_for_turboquant(model)
        del cache
        clear_vram()

def run_model_suite(model_id):
    print(f"\n🚀 Testing Model: {model_id}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map={"": 0}, 
        trust_remote_code=True
    )
    model.eval()
    
    model_res = {}
    
    # Prompt for qualitative check
    PROMPT = "The fundamental concept of Quantum Entanglement is"
    ids_small = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda")

    for ctx in CONTEXT_SIZES:
        print(f"  --- Context: {ctx} tokens ---")
        # Build long dummy context + real prompt
        long_ids = torch.randint(0, tokenizer.vocab_size, (1, ctx - ids_small.shape[1]), device="cuda")
        ids = torch.cat([long_ids, ids_small], dim=1)
        
        ctx_res = {}
        
        # Baseline
        print("    Measuring Baseline...")
        b = measure_step(model, tokenizer, ids, label="Baseline")
        ctx_res["baseline"] = b
        
        # TQ 4-bit
        print("    Measuring TurboQuant 4-bit...")
        t4 = measure_step(model, tokenizer, ids, bits=4, label="TQ4b")
        ctx_res["tq4b"] = t4
        
        # TQ 3-bit
        print("    Measuring TurboQuant 3-bit...")
        t3 = measure_step(model, tokenizer, ids, bits=3, label="TQ3b")
        ctx_res["tq3b"] = t3
        
        # Accuracies vs Baseline
        if b and t4:
            cos = F.cosine_similarity(b["logits"], t4["logits"]).mean().item()
            t4["cossim"] = cos
        if b and t3:
            cos = F.cosine_similarity(b["logits"], t3["logits"]).mean().item()
            t3["cossim"] = cos
            
        model_res[ctx] = ctx_res
        
    del model, tokenizer
    clear_vram()
    return model_res

if __name__ == "__main__":
    for mid in MODELS:
        try:
            results[mid] = run_model_suite(mid)
        except Exception as e:
            print(f"Failed to test {mid}: {e}")
            
    # Save results to JSON
    with open("bench_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: str(x) if isinstance(x, torch.Tensor) else None)
    print("\n✅ Benchmark results saved to bench_results.json")
