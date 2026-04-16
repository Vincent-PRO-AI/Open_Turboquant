
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from tq_impl import TurboQuantCache, patch_model_for_turboquant, unpatch_model_for_turboquant

# Using the larger 26B version
MODEL_ID = "google/gemma-4-26B-A4B"

def get_total_vram():
    total = 0
    for i in range(torch.cuda.device_count()):
        torch.cuda.empty_cache()
        torch.cuda.synchronize(i)
        total += torch.cuda.memory_allocated(i)
    return total / 1024**3

def incremental_prefill(model, input_ids, cache, chunk_size=2048):
    seq_len = input_ids.shape[1]
    for i in range(0, seq_len, chunk_size):
        end = min(i + chunk_size, seq_len)
        chunk = input_ids[:, i:end]
        with torch.no_grad():
            model(chunk, past_key_values=cache, use_cache=True)
        if i % 8192 == 0:
            print(f"    Processed {end}/{seq_len} tokens...", flush=True)

def run_large_model_benchmark():
    print(f"=== TurboQuant Real-World Benchmark (Gemma-4-26B FP16) ===")
    
    # We load in FP16 and distribute across both GPUs (40GB total)
    # 26B model in FP16 = ~33.3 GB
    print(f"Loading {MODEL_ID} in FP16 across both GPUs...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    base_vram = get_total_vram()
    print(f"Base Model VRAM: {base_vram:.2f} GB (Total)")
    
    # Target Contexts
    TARGETS = [8192, 16384, 32768, 65536]
    
    first_device = next(model.parameters()).device
    
    print("\n{:>10} | {:>12} | {:>14} | {:>16} | {:>8}".format(
        "Context", "KV VRAM (G)", "Prefill (t/s)", "Decode (t/s)", "Ratio"))
    print("-" * 75)
    
    for ctx in TARGETS:
        text = "Deep benchmark text. " * (ctx // 4)
        ids = tokenizer(text, return_tensors="pt", max_length=ctx, truncation=True).input_ids.to(first_device)
        actual_len = ids.shape[1]
        
        # 4-bit Keys and 4-bit Values
        cache = TurboQuantCache(bits=4.0, bits_value=4.0, dtype=torch.float16)
        patch_model_for_turboquant(model, cache)
        
        try:
            # Measure Prefill
            t0 = time.perf_counter()
            incremental_prefill(model, ids, cache)
            t_prefill = time.perf_counter() - t0
            
            # Measure Decode
            q = torch.randint(0, 100, (1, 1), device=first_device)
            t0 = time.perf_counter()
            n_steps = 5
            for _ in range(n_steps):
                with torch.no_grad():
                    model(q, past_key_values=cache, use_cache=True)
            t_decode = (time.perf_counter() - t0) / n_steps
            
            v_total = get_total_vram()
            kv_vram = v_total - base_vram
            stats = cache.memory_footprint()
            ratio = stats.get('key_compression_ratio', 0.0)
            
            print("{:>10} | {:>12.2f} | {:>14.1f} | {:>16.1f} | {:>7.1f}x".format(
                actual_len, kv_vram, actual_len/t_prefill, 1.0/t_decode, ratio))
                
        except torch.cuda.OutOfMemoryError:
            print("{:>10} | {:>12} | {:>14} | {:>16} | {:>8}".format(
                actual_len, "OOM", "-", "-", "-"))
        except Exception as e:
            print(f"  Error at {ctx}: {e}")
            
        unpatch_model_for_turboquant(model)
        cache.reset()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_large_model_benchmark()
