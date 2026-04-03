import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tq_impl import TurboQuantCache, patch_model_for_turboquant

torch.set_grad_enabled(False)

print("Loading model...")
model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", torch_dtype=torch.float16)
model.eval()

prompt = "Explain in detail the nature of the universe and how quantum" * 4
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
ids = inputs["input_ids"]

print("Prompt length:", ids.shape[1])

# Baseline generate
gb = model.generate(ids, max_new_tokens=4, do_sample=False, return_dict_in_generate=True, output_logits=True)

# TurboQuant cache generate
c2 = TurboQuantCache(bits=4.0, dtype=torch.float16)
gt = model.generate(ids, past_key_values=c2, max_new_tokens=4, do_sample=False, return_dict_in_generate=True, output_logits=True)

print("Baseline generated:", tokenizer.decode(gb.sequences[0, ids.shape[1]:]))
print("TurboQuant generated:", tokenizer.decode(gt.sequences[0, ids.shape[1]:]))

for i in range(4):
    l_b = gb.logits[i]
    l_t = gt.logits[i]
    cos = F.cosine_similarity(l_b, l_t, dim=-1).mean().item()
    t1 = (l_b.argmax(-1) == l_t.argmax(-1)).float().mean().item()
    print(f"Step {i}: cos={cos:.4f}, top1={t1:.1%}")
