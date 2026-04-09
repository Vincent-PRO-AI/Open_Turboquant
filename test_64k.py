import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tq_impl import TurboQuantCache, patch_model_for_turboquant
import time

MODEL_ID = "google/gemma-4-E2B-it"
CONTEXTS = [16384, 32768, 65536] 

def get_vram():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**3

print(f"--- Loading {MODEL_ID} with Flash Attention 2 ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="cuda:0", 
        dtype=torch.float16, 
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
except Exception as e:
    print(f"Flash Attention 2 not available ({e}), falling back to standard...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="cuda:0", 
        dtype=torch.float16, 
        trust_remote_code=True
    )
model.eval()

base_vram = get_vram()
print(f"Base VRAM (Model): {base_vram:.2f} GB")

results = []

for ctx in CONTEXTS:
    print(f"\n[Target Context {ctx}]")
    # Repetition to reach context
    text = "Ceci est un test de contexte colossal pour TurboQuant V2. " * (ctx // 10)
    ids = tokenizer(text, return_tensors="pt", max_length=ctx, truncation=True).input_ids.to("cuda:0")
    actual_len = ids.shape[1]
    print(f"  Actual tokens: {actual_len}")
    
    # 1. Baseline FP16
    torch.cuda.empty_cache()
    try:
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(ids, max_new_tokens=1, do_sample=False, use_cache=True)
        dt = time.perf_counter() - t0
        v_total = get_vram()
        kv_vram_fp16 = v_total - base_vram
        print(f"  FP16: {kv_vram_fp16:.2f} GB (Total: {v_total:.2f} GB, Time: {dt:.2f}s)")
    except Exception as e:
        print(f"  FP16: OOM / Error ({type(e).__name__})")
        kv_vram_fp16 = float('nan')

    # 2. TurboQuant 4-bit
    torch.cuda.empty_cache()
    cache = TurboQuantCache(bits=4.0)
    patch_model_for_turboquant(model, cache)
    try:
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(ids, past_key_values=cache, max_new_tokens=1, do_sample=False, use_cache=True)
        dt = time.perf_counter() - t0
        v_total = get_vram()
        kv_vram_tq = v_total - base_vram
        stats = cache.memory_footprint()
        print(f"  TQ 4-bit: {kv_vram_tq:.2f} GB (Total: {v_total:.2f} GB, Time: {dt:.2f}s)")
        print(f"  TQ Ratio: {stats['key_compression_ratio']:.1f}x")
        
        results.append({'ctx': actual_len, 'fp16': kv_vram_fp16, 'tq': kv_vram_tq, 'ratio': stats['key_compression_ratio']})
    except Exception as e:
        print(f"  TQ 4-bit: OOM / Error ({type(e).__name__})")
    
    from tq_impl import unpatch_model_for_turboquant
    unpatch_model_for_turboquant(model)

print("\n" + "="*50)
print("FINAL RESULTS: 64K CONTEST")
print("="*50)
print(f"{'Context':>8} | {'FP16 (GB)':>10} | {'TQ 4b (GB)':>10} | {'Ratio':>6}")
for r in results:
    print(f"{r['ctx']:>8} | {r['fp16']:>10.2f} | {r['tq']:>10.2f} | {r['ratio']:>6.1f}x")
