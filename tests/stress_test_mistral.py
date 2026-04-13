import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tq_impl import patch_model_for_turboquant, TurboQuantCache
import time

def test_mistral_production():
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"[INFO] Loading Production Model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Use device_map='auto' to stress multi-GPU setup (RTX 5080 + 4090)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    print("[INFO] Patching Mistral with TurboQuant...")
    patch_model_for_turboquant(model)
    
    # Initialize the compressed cache
    print("[INFO] Initializing TurboQuantCache (bits=4)...")
    cache = TurboQuantCache(bits=4, max_seq_len=8192, dtype=torch.float16)
    
    prompt = "Explain the importance of KV cache compression in large language models in detail."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("[INFO] Starting Generation with Compressed Cache...")
    start_time = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs, 
            past_key_values=cache,
            max_new_tokens=150, 
            do_sample=True, 
            temperature=0.7,
            use_cache=True
        )
    end_time = time.time()
    
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n================ MODEL RESPONSE ================")
    print(decoded)
    print("================================================")
    
    print(f"\n[STATS] Time taken: {end_time - start_time:.2f}s")
    print(f"[STATS] Tensors on devices: {model.hf_device_map}")
    print(f"[STATS] Final Cache Length: {cache.get_seq_length()} tokens")
    print("[SUCCESS] Mistral-7B Validation Complete!")

if __name__ == "__main__":
    test_mistral_production()
