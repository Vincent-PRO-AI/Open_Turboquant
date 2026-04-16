import gc
import math
import os
import sys
import time
from typing import Dict, List, Optional

import psutil
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Ensure tq_impl is in path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

def get_vram_gb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated(0) / 1024**3, torch.cuda.max_memory_allocated(0) / 1024**3

def get_ram_gb():
    return psutil.Process().memory_info().rss / 1024**3

def safe_import_tq():
    """Try to import TQ from different possible structures (v2 vs legacy)."""
    try:
        # v2 (Current)
        from tq_impl.cache import TurboQuantCache
        from tq_impl.model_patch import patch_model_for_turboquant, unpatch_model_for_turboquant
        return TurboQuantCache, patch_model_for_turboquant, unpatch_model_for_turboquant
    except (ImportError, ModuleNotFoundError):
        try:
            # legacy (main-legacy)
            from tq_impl import TurboQuantCache, patch_model_for_turboquant, unpatch_model_for_turboquant
            return TurboQuantCache, patch_model_for_turboquant, unpatch_model_for_turboquant
        except (ImportError, ModuleNotFoundError) as e:
            print(f"    [ERROR] Fatal import failure: {e}")
            return None, None, None

class AuditGemma:
    def __init__(self, model_id: str, label: str = "v2"):
        self.model_id = model_id
        self.label = label
        self.results = {}
        
        print(f"\n[Audit] Loading {model_id} on RTX 4090 (Label: {label})")
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map={"": 0},
            quantization_config=quant_config,
            trust_remote_code=True
        )
        self.model.eval()
        
    def run_test(self, name: str, prompt: str, max_new_tokens: int = 64, use_tq: bool = True, fused: bool = False):
        print(f"  > Running: {name} (TQ={use_tq}, Fused={fused})")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        compute_dtype = next(self.model.parameters()).dtype
        
        cache = None
        if use_tq:
            TQCache, patch_fn, unpatch_fn = safe_import_tq()
            if TQCache is None:
                use_tq = False
            else:
                cache = TQCache(bits=4.0, dtype=compute_dtype)
                if fused:
                    patch_fn(self.model, cache)

        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    past_key_values=cache,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True
                )
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            
            # Clean up patch
            if fused and use_tq:
                unpatch_fn(self.model)
                
            v_now, v_peak = get_vram_gb()
            ram = get_ram_gb()
            
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            n_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            tps = n_tokens / dt if dt > 0 else 0
            
            print(f"    Result: {tps:.2f} tok/s | VRAM Peak: {v_peak:.2f} GB | RAM: {ram:.2f} GB")
            
            return {
                "tps": tps,
                "vram_peak": v_peak,
                "ram_gb": ram,
                "text": text,
                "n_tokens": n_tokens
            }
        except torch.cuda.OutOfMemoryError:
            print("    [ERROR] Out of Memory!")
            if fused:
                unpatch_model_for_turboquant(self.model)
            return {"error": "OOM"}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="v2")
    parser.add_argument("--skip_31b", action="store_true")
    args = parser.parse_args()

    # Force 4090 only
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 1. Quality Test (2B)
    audit_2b = AuditGemma("google/gemma-4-E2B-it", label=args.label)
    prompts = [
        "Explain the difference between L1 and L2 normalization in KV cache quantization.",
        "Write a short poem about the speed of light.",
        "If a model has 8 layers and each layer takes 2ms, how long does the full forward pass take?"
    ]
    
    res_2b = {"baseline": [], "tq": [], "tq_fused": []}
    
    for p in prompts:
        res_2b["baseline"].append(audit_2b.run_test("Quality 2B", p, use_tq=False))
        res_2b["tq"].append(audit_2b.run_test("Quality 2B", p, use_tq=True, fused=False))
        res_2b["tq_fused"].append(audit_2b.run_test("Quality 2B", p, use_tq=True, fused=True))

    del audit_2b
    gc.collect()
    torch.cuda.empty_cache()

    if not args.skip_31b:
        # 2. Stress Test (31B)
        print("\n" + "="*50)
        print("STRESS TEST: GEMMA-4 31B")
        print("="*50)
        
        audit_31b = AuditGemma("google/gemma-4-31B-it", label=args.label)
        # Massive context simulation (repetition of a prompt)
        long_prompt = "Summarize the following text: " + ("Large scale language models are changing the world. " * 50) # Approx 500 tokens
        
        # Test baseline first (might OOM)
        audit_31b.run_test("Stress 31B", long_prompt, max_new_tokens=128, use_tq=False)
        # Test TQ fused
        audit_31b.run_test("Stress 31B", long_prompt, max_new_tokens=128, use_tq=True, fused=True)
    
    # Final Summary (Print to console, I'll capture it)
    print("\n--- AUDIT FINAL ---")
    print(f"Mode: {os.environ.get('TQ_LOG_MODE', 'unknown')}")
    # ... rest of summary logic ...

if __name__ == "__main__":
    main()
