import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from tq_impl.cache import TurboQuantCache
from tq_impl.model_patch import patch_model_for_turboquant as patch_model_with_tq

def run_sweep(model_id="google/gemma-2-2b-it", bits_list=[3.0, 4.0], context_list=[512, 1024]):
    print(f"Starting sweep for {model_id}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    
    # Simple prompt
    text = "Explain the importance of KV cache compression in large language models."
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    results = []
    
    for bits in bits_list:
        for ctx in context_list:
            print(f"\n--- Testing bits={bits}, ctx={ctx} ---")
            
            # Create TQ cache
            cache = TurboQuantCache(bits=bits)
            patch_model_with_tq(model)
            
            # Warmup / Prefill
            start_time = time.time()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    past_key_values=cache,
                    max_new_tokens=ctx,
                    do_sample=False,
                    use_cache=True,
                )
            end_time = time.time()
            
            duration = end_time - start_time
            tps = ctx / duration
            
            mem = cache.memory_footprint()
            ratio = mem["key_compression_ratio"]
            
            print(f"Speed: {tps:.2f} tok/s")
            print(f"Compression Ratio: {ratio:.2f}x")
            
            results.append({
                "bits": bits,
                "ctx": ctx,
                "tps": tps,
                "ratio": ratio
            })
            
            # Reset for next run
            cache.reset()
            
    print("\nSweep Results Summary:")
    print("Bits | Ctx | Speed (tok/s) | Compression")
    print("-" * 45)
    for r in results:
        print(f"{r['bits']:.1f}  | {r['ctx']:4} | {r['tps']:12.2f} | {r['ratio']:10.2f}x")

if __name__ == "__main__":
    # Small test on Gemma 2B
    run_sweep()
