import os
import sys
import torch
import time
import argparse
from typing import List

# Enable import of tq_impl
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tq_impl import TurboQuantCache, patch_model_for_turboquant

def print_row(tokens, vram, status="Active"):
    print(f"| {tokens:8} | {vram:9.2f} GB | {status:10} |")

def run_scaling_benchmark(model_id="google/gemma-4-31B-it", token=None, use_tq=True, max_tokens=65536, chunk_size=4096):
    mode = "TURBOQUANT (4-bit KV)" if use_tq else "BASELINE (BF16 KV)"
    print("\n" + "="*60)
    print(f"🏃 RUNNING BENCHMARK: {mode}")
    print("="*60)
    print(f"| Tokens   | VRAM Peak | Status     |")
    print(f"|----------|-----------|------------|")

    # 1. Load Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=token
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Setup Cache
    cache = None
    if use_tq:
        cache = TurboQuantCache(
            bits_key=4.0, bits_value=8.0, 
            outliers=True, dtype=model.dtype,
            max_seq_len=max_tokens + 1024
        )
        patch_model_for_turboquant(model, cache)

    # 3. Scaling Loop
    dummy_input = torch.randint(0, 1000, (1, chunk_size), device=model.device)
    total_tokens = 0
    past_key_values = cache if use_tq else None
    
    try:
        while total_tokens < max_tokens:
            torch.cuda.reset_peak_memory_stats()
            
            with torch.inference_mode():
                # Perform one forward pass with the chunk
                outputs = model(
                    dummy_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                
                # Update past_key_values for next iteration
                if not use_tq:
                    past_key_values = outputs.past_key_values
                else:
                    # In TQ, the cache object is updated in-place during patching
                    pass
            
            total_tokens += chunk_size
            vram_peak = torch.cuda.max_memory_allocated() / 1024**3
            print_row(total_tokens, vram_peak)
            
            if vram_peak > 47.5:
                print("⚠️  Warning: Near Blackwell VRAM Limit!")
                break

    except torch.cuda.OutOfMemoryError:
        print_row(total_tokens, torch.cuda.max_memory_allocated() / 1024**3, "💥 OOM!")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Cleanup for next run
    del model
    del tokenizer
    if cache: del cache
    torch.cuda.empty_cache()
    time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--use_tq", action="store_true", help="Enable TurboQuant")
    args = parser.parse_args()

    # Run selected benchmark
    run_scaling_benchmark(args.model, args.token, use_tq=args.use_tq, max_tokens=args.max_tokens, chunk_size=args.chunk_size)
