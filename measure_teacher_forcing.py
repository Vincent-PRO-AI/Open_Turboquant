import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tq_impl import TurboQuantCache

model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", torch_dtype=torch.float16)

prompt = "Explain in detail the nature of the universe"
ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

with torch.inference_mode():
    n_dec = 8
    
    # Baseline decode
    gb = model.generate(ids, max_new_tokens=n_dec, do_sample=False, return_dict_in_generate=True, output_logits=True)
    baseline_seq = gb.sequences
    
    from transformers import DynamicCache
    c2 = DynamicCache()
    
    # Prefill
    out = model(ids, past_key_values=c2, use_cache=True)
    gt_logits.append(out.logits[:, -1, :])
    
    # Decode step by step using EXACT baseline tokens
    for i in range(1, n_dec):
        next_tok = baseline_seq[:, ids.shape[1] + i - 1].unsqueeze(-1)
        out = model(next_tok, past_key_values=c2, use_cache=True)
        gt_logits.append(out.logits[:, -1, :])
    
    cos_d, top1_d = [], []
    for i in range(n_dec):
        l_b = gb.logits[i]
        l_t = gt_logits[i]
        cos = F.cosine_similarity(l_b, l_t, dim=-1).mean().item()
        top1 = (l_b.argmax(-1) == l_t.argmax(-1)).float().mean().item()
        print(f"Step {i}: cos={cos:.4f}, top1={top1:.1%}")
