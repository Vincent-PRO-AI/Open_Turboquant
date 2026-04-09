import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from tq_impl.cache import TurboQuantCache
from tq_impl.patch import patch_model_with_tq

def layer_sweep(model_id="google/gemma-2-2b-it"):
    print(f"Starting layer-specific sweep for {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    num_layers = model.config.num_hidden_layers
    
    text = "Explain the importance of KV cache compression."
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Strategy 1: All 4 bits
    # Strategy 2: All 3 bits
    # Strategy 3: Half-and-half (First half 4b, Second half 3b)
    # Strategy 4: Outlier-heavy (First 2 layers FP16, rest 3b)
    
    strategies = {
        "Baseline (4b)": 4.0,
        "Extreme (3b)": 3.0,
        "Hybrid (1/2 4b, 1/2 3b)": {i: (4.0 if i < num_layers // 2 else 3.0) for i in range(num_layers)},
        "Outlier-Safe (L0-2 FP16, rest 3b)": {i: (4.0 if i < 3 else 3.0) for i in range(num_layers)},
    }
    
    patch_model_with_tq(model)
    
    print("\nStrategy | Speed (tok/s) | Compression | Ratio vs FP16")
    print("-" * 65)
    
    for name, config in strategies.items():
        cache = TurboQuantCache(bits=config)
        
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, past_key_values=cache, max_new_tokens=256, do_sample=False)
        torch.cuda.synchronize()
        duration = time.time() - start
        
        mem = cache.memory_footprint()
        ratio = mem["key_compression_ratio"]
        tps = 256 / duration
        
        print(f"{name:25} | {tps:12.2f} | {ratio:10.2f}x")
        cache.reset()

if __name__ == "__main__":
    layer_sweep()
