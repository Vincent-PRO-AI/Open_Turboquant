from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it", torch_dtype=torch.float16, device_map="cuda:0")
tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
ids = tok("hello world", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    out = model(ids, use_cache=True)
pv = out.past_key_values
print("Type:", type(pv).__name__)
print("Attrs:", [a for a in dir(pv) if not a.startswith("_")])
