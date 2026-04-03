import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", torch_dtype=torch.float16)

prompt = "Explain in detail the nature of the universe"
ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

with torch.inference_mode():
    out = model(ids, output_hidden_states=True)
    hs = out.hidden_states[10]  # Let's pick layer 10
    
    # Let's project to Q and K
    attn = model.model.layers[10].self_attn
    q = attn.q_proj(hs)
    k = attn.k_proj(hs)
    
    B, T, D = q.shape
    q = q.view(B, T, -1, 128)
    k = k.view(B, T, -1, 128)
    
    q_norm = q.norm(dim=-1).max().item()
    k_norm = k.norm(dim=-1).max().item()
    print("Max Q norm:", q_norm)
    print("Max K norm:", k_norm)
    print("Min Q norm:", q.norm(dim=-1).min().item())
    print("Min K norm:", k.norm(dim=-1).min().item())
