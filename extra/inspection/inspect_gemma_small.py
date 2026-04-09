import torch
from transformers import AutoModelForCausalLM

model_id = "google/gemma-4-E2B-it"
print(f"Inspecting {model_id}...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    print("Model loaded.")
    for name, module in model.named_modules():
        if "attn" in name.lower() or "attention" in name.lower():
            print(f"Layer: {name} | Class: {type(module).__name__}")
            # Break after first few to save output
            if "layers.2" in name:
                break
except Exception as e:
    print(f"Error: {e}")
