import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tq_impl import TurboQuantCache, patch_model_for_turboquant
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-4-E2B-it"
BITS = 4.0

print(f"--- Loading {MODEL_ID} ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", dtype=torch.float16, trust_remote_code=True
)
model.eval()

# ---------------------------------------------------------------------------
# Interaction Loop
# ---------------------------------------------------------------------------
print(f"\n--- TurboQuant V2 Playground Activated ---")
print(f"Mode: {BITS}-bit compression (with Outlier Retention)")
print("Type 'exit' to quit.\n")

while True:
    prompt = input("User: ")
    if prompt.lower() in ["exit", "quit"]:
        break
    
    msgs = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if hasattr(ids, "input_ids"): ids = ids.input_ids
    ids = ids.to(next(model.parameters()).device)
    
    # 1. Standard Generation (FP16 Baseline)
    print("\n[Baseline FP16] Thinking...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        out_base = model.generate(ids, max_new_tokens=100, do_sample=True, temperature=0.7)
    dt_base = time.perf_counter() - t0
    print(f"Assistant: {tokenizer.decode(out_base[0][ids.shape[1]:], skip_special_tokens=True)}")
    print(f"Cost: {dt_base:.2f}s")
    
    # 2. TurboQuant Generation
    print(f"\n[TurboQuant {BITS}b] Thinking...")
    cache = TurboQuantCache(bits=BITS)
    # Enable fused kernels for faster decode
    patch_model_for_turboquant(model, cache)
    
    t0 = time.perf_counter()
    with torch.inference_mode():
        out_tq = model.generate(ids, past_key_values=cache, max_new_tokens=100, do_sample=True, temperature=0.7)
    dt_tq = time.perf_counter() - t0
    
    # Unpatch for next baseline run
    from tq_impl import unpatch_model_for_turboquant
    unpatch_model_for_turboquant(model)
    
    print(f"Assistant: {tokenizer.decode(out_tq[0][ids.shape[1]:], skip_special_tokens=True)}")
    print(f"Cost: {dt_tq:.2f}s")
    
    # VRAM stats
    mem = cache.memory_footprint()
    print(f"VRAM Savings: {mem['key_compression_ratio']:.1f}x (KV Cache only)")
    print("-" * 50)
