
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tq_impl.cache import TurboQuantCache
from tq_impl.model_patch import patch_model_for_turboquant

def diag_gemma_pipeline():
    model_id = "google/gemma-4-E2B-it" # Use the model already in cache
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu") # Start with CPU to avoid VRAM issues
    model = model.to('cuda')
    
    cache = TurboQuantCache(bits=4.0)
    patch_model_for_turboquant(model, cache)
    
    text = "Hello, how are you today?"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    print("Running generate...")
    try:
        with torch.no_grad():
            # Prefill + first few tokens of decode
            output = model.generate(**inputs, past_key_values=cache, max_new_tokens=5, use_cache=True)
        print("Success! Generated output.")
        print(f"Decoded: {tokenizer.decode(output[0])}")
    except Exception as e:
        print(f"Error during generate: {e}")
        import traceback
        traceback.print_exc()

    # Check for NaNs in the internal cache
    for li, fr in cache._final_radii.items():
        if torch.isnan(fr).any():
            print(f"  !! Layer {li}: NaNs found in Radii!")
    
    for li, kr in cache._sketched_buffer.items():
        if torch.isnan(kr).any():
            print(f"  !! Layer {li}: NaNs found in Sketched Buffer!")

if __name__ == "__main__":
    diag_gemma_pipeline()
