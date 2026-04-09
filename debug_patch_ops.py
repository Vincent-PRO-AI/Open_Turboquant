import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tq_impl import TurboQuantCache, patch_model_for_turboquant
import tq_impl.model_patch as mp

# Modify mp to log calls
original_fused = mp._fused_decode
def debug_fused(*args, **kwargs):
    print(f"[DEBUG] _fused_decode called for layer {args[4]}")
    return original_fused(*args, **kwargs)
mp._fused_decode = debug_fused

model_id = "google/gemma-4-E2B-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

prompt = "What is the capital of France?"
msgs = [{"role": "user", "content": prompt}]
ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
if hasattr(ids, "input_ids"): ids = ids.input_ids
ids = ids.to(next(model.parameters()).device)

cache = TurboQuantCache(bits=4.0)
patch_model_for_turboquant(model, cache)

print("\n--- Starting Generate ---")
with torch.inference_mode():
    out = model.generate(ids, past_key_values=cache, max_new_tokens=20, do_sample=False)
print("--- End Generate ---")

print(f"Generated text: {tokenizer.decode(out[0], skip_special_tokens=True)}")
print(f"Final cache seq len: {cache.get_seq_length(0)}")
print(f"Memory footprint: {cache.memory_footprint()}")
