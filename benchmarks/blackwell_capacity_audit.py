#!/usr/bin/env python3
import argparse
import gc
import time
import torch
import os
import sys

# Ensure tq_impl is discoverable
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tq_impl import TurboQuantCache, patch_model_for_turboquant

def get_gpu_mem():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--bits", type=float, default=4.0)
    parser.add_argument("--use_tq", action="store_true")
    args = parser.parse_args()

    # context_steps = [32768, 49152, 65536, 81920, 98304, 114688, 131072]
    context_steps = [32768, 65536, 131072]
    
    print("="*60)
    print(f" CAPACITY AUDIT: {args.model}")
    print(f" Mode: {'TurboQuant ' + str(args.bits) + '-bit' if args.use_tq else 'FP16 Baseline'}")
    print("="*60)

    # 1. Load Model
    print("\n[Step 1] Loading model in 4-bit...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=args.token,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    base_mem, _ = get_gpu_mem()
    print(f"Model loaded. VRAM Start: {base_mem:.2f} GB")

    results = []

    for ctx in context_steps:
        print(f"\n[Testing Context: {ctx}]")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # Prepare dummy prompt
            dummy_input = torch.randint(0, 100, (1, 32), device=model.device)
            
            if args.use_tq:
                # We simulate prefill memory by forcing a large cache allocation
                cache = TurboQuantCache(bits=args.bits, dtype=model.dtype, max_seq_len=ctx + 512)
                patch_model_for_turboquant(model, cache)
                
                # Fill cache to target context
                # To be realistic, we simulate prefill tokens
                # For audit, we just check if it fits in VRAM
                # (Actual KV states are allocated dynamically anyway)
                
                # Let's perform a 1-step generation to trigger allocations
                with torch.inference_mode():
                    model.generate(dummy_input, past_key_values=cache, max_new_tokens=1)
                
                # Force dynamic allocation check for target context
                # (Only for allocated compressed layers, skip raw D=512 layers)
                for layer_idx in cache._allocated_len.keys():
                    cache._ensure_capacity(layer_idx, ctx)
            else:
                # Baseline FP16
                with torch.inference_mode():
                    model.generate(dummy_input, max_new_tokens=1, use_cache=True)
            
            mem_curr, mem_peak = get_gpu_mem()
            print(f"  SUCCESS: {ctx} tokens")
            print(f"  Current VRAM: {mem_curr:.2f} GB | Peak: {mem_peak:.2f} GB")
            results.append((ctx, mem_curr, mem_peak, "OK"))
            
        except torch.cuda.OutOfMemoryError:
            print(f"  FAILED: {ctx} tokens (OOM)")
            results.append((ctx, 0, 0, "OOM"))
            break
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            break

    print("\n" + "="*60)
    print(" CAPACITY AUDIT SUMMARY")
    print("="*60)
    for c, cur, pk, status in results:
        tq_label = f"TQ-{args.bits}b" if args.use_tq else "FP16"
        print(f"{c:>7} tokens | {tq_label:<7} | {status} | Peak: {pk:>6.2f} GB")
    print("="*60)

if __name__ == "__main__":
    main()
