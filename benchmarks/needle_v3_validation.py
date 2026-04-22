#!/usr/bin/env python3
import argparse
import os
import sys
import torch
import random

# Ensure tq_impl is discoverable
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tq_impl import TurboQuantCache, patch_model_for_turboquant

def create_needle_haystack(tokenizer, context_size, needle_pos_pct=0.5):
    needle = "Le mot secret de la certification TurboQuant est 'DIAMANT-BLACKWELL'."
    filler = "Le cache KV est une structure de données essentielle pour l'inférence efficace des modèles de langage. "
    
    # Estimate tokens per filler sentence
    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    num_fillers = (context_size // len(filler_tokens)) + 1
    
    needle_idx = int(num_fillers * needle_pos_pct)
    
    haystack = []
    for i in range(num_fillers):
        if i == needle_idx:
            haystack.append(needle)
        haystack.append(filler)
        
    full_text = " ".join(haystack)
    prompt = f"Voici un long document technique :\n\n{full_text}\n\nQuestion : Quel est le mot secret de la certification TurboQuant ? Réponse : Le mot secret est '"
    
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--ctx", type=int, default=32768)
    parser.add_argument("--pos", type=float, default=0.7) # Place needle at 70% depth
    parser.add_argument("--bits", type=float, default=4.0)
    args = parser.parse_args()

    print(f"Loading {args.model} for Retrieval Test ({args.ctx} tokens)...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=args.token,
        quantization_config=quantization_config,
        device_map="auto"
    )

    prompt = create_needle_haystack(tokenizer, args.ctx, args.pos)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    actual_ctx = inputs.input_ids.shape[1]
    
    print(f"Haystack ready. Total tokens: {actual_ctx}")
    print(f"Needle inserted at ~{args.pos*100}% depth.")

    # Prepare chunks for prefill to avoid SDPA OOM (128GB+ spike)
    # We process in chunks of 2048 tokens
    CHUNK_SIZE = 2048
    input_ids = inputs.input_ids
    T_total = input_ids.shape[1]
    
    cache = TurboQuantCache(bits=args.bits, dtype=model.dtype, max_seq_len=T_total + 64)
    patch_model_for_turboquant(model, cache)
    
    print(f"\n--- RETRIEVAL TEST (NEEDLE) ---")
    print(f"Phase: Chunked Prefill ({T_total} tokens)...")
    
    with torch.inference_mode():
        for start_idx in range(0, T_total - 1, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, T_total - 1)
            chunk = input_ids[:, start_idx:end_idx]
            # Standard forward to fill cache
            model(chunk, past_key_values=cache, use_cache=True)
            if start_idx % 8192 == 0:
                print(f"  Processed {end_idx}/{T_total} tokens...")

    print("Phase: Generation (Retrieving Needle)...")
    with torch.inference_mode():
        # Last token and generate
        last_token = input_ids[:, -1:]
        outputs = model.generate(
            last_token,
            past_key_values=cache,
            max_new_tokens=15,
            do_sample=False
        )
    
    generated_text = tokenizer.decode(outputs[0, actual_ctx:], skip_special_tokens=True)
    print(f"Model Output: '{generated_text}'")
    
    success = "DIAMANT-BLACKWELL" in generated_text
    print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print("-------------------------------")

if __name__ == "__main__":
    main()
