import os
import sys
import torch
import time
import argparse

# Enable import of tq_impl
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tq_impl import TurboQuantCache, patch_model_for_turboquant

def run_test(model_id="google/gemma-4-31B-it", token=None):
    print("=" * 80)
    print(f"🚀 GEMMA-4 31B STABILIZATION TEST (RTX 4090 24GB)")
    print("=" * 80)

    # 1. Load in 4-bit weights (Mandatory for 31B on 24GB)
    print(f"\n[1/3] Loading 4-bit quantized weights for {model_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=token
        )
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return

    # 2. Patch with TurboQuant Elite V3
    print(f"\n[2/3] Initializing TurboQuant Elite V3 (4-bit KV)...")
    cache = TurboQuantCache(
        bits_key=4.0, 
        bits_value=8.0, 
        outliers=True, 
        dtype=model.dtype  # Match model (BFloat16)
    )
    patch_model_for_turboquant(model, cache)
    print("✅ Model patched and ready.")

    # 3. Validation Prompt
    prompt = "Explain the architecture of the Blackwell GPU and how it interacts with Tensor Cores."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"\n[3/3] Generating (256 tokens)...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False, # Deterministic for parity check
            past_key_values=cache
        )
    
    t1 = time.time()
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    vram_peak = torch.cuda.max_memory_allocated() / 1024**3

    print("\n" + "=" * 80)
    print("MODEL RESPONSE:")
    print("-" * 80)
    print(response[len(prompt):].strip())
    print("=" * 80)
    
    print(f"\n📊 RESULTS:")
    print(f"  - Generated Tokens: {len(outputs[0]) - inputs.input_ids.shape[1]}")
    print(f"  - Speed: {(len(outputs[0]) - inputs.input_ids.shape[1]) / (t1 - t0):.2f} tokens/s")
    print(f"  - VRAM Peak: {vram_peak:.2f} GB / 24.00 GB")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    args = parser.parse_args()
    run_test(args.model, args.token)
