import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", torch_dtype=torch.float16)

prompt = "Explain in detail the nature of the universe"
ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

with torch.inference_mode():
    out = model(ids, output_hidden_states=True)
    hs = out.hidden_states[10]
    
    attn = model.model.layers[10].self_attn
    k = attn.k_proj(hs)
    B, T, D = k.shape
    k = k.view(B, T, -1, 128)
    
    k_mean = k.mean(dim=1, keepdim=True)
    k_centered = k - k_mean
    
    print("Max K norm before centering:", k.norm(dim=-1).max().item())
    print("Max K norm after centering: ", k_centered.norm(dim=-1).max().item())
    print("Mean K norm after centering:", k_centered.norm(dim=-1).mean().item())
