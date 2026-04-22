#!/usr/bin/env python3
import argparse
import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

# Ensure tq_impl is discoverable
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tq_impl import TurboQuantCache, patch_model_for_turboquant, unpatch_model_for_turboquant

def evaluate_ppl(model, tokenizer, dataset_text, bits, use_tq=False, max_length=2048, stride=512):
    encodings = tokenizer(dataset_text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    # Optional: Patch model
    cache = None
    if use_tq:
        cache = TurboQuantCache(bits=bits, dtype=model.dtype, max_seq_len=max_length + stride)
        patch_model_for_turboquant(model, cache)
    
    print(f"Evaluating PPL (TQ={use_tq}, bits={bits})...")

    try:
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # how many new tokens to calculate loss for
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                # Note: TurboQuantCache currently handles internal state. 
                # For a sliding window PPL, we reset cache each time or manage it.
                # To be safe and independent for each window:
                if use_tq:
                    current_cache = TurboQuantCache(bits=bits, dtype=model.dtype, max_seq_len=max_length + stride)
                    # We need to re-patch or update the weakref if we use a new cache object
                    patch_model_for_turboquant(model, current_cache)
                    outputs = model(input_ids, labels=target_ids, past_key_values=current_cache)
                else:
                    outputs = model(input_ids, labels=target_ids)

                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood * trg_len)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
    finally:
        if use_tq:
            unpatch_model_for_turboquant(model)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--bits", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=1) # Just a few windows for faster audit
    args = parser.parse_args()

    # Load Model
    print(f"Loading {args.model} in 4-bit...")
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

    # Load Dataset (Wikitext-2 subset)
    from datasets import load_dataset
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset_text = "\n\n".join(test["text"][:1000]) # Use first 1000 lines for a quick audit

    print("\n--- PERPLEXITY AUDIT ---")
    
    # 1. Baseline
    ppl_base = evaluate_ppl(model, tokenizer, dataset_text, bits=16.0, use_tq=False)
    print(f"Baseline PPL: {ppl_base:.4f}")

    # 2. TurboQuant
    ppl_tq = evaluate_ppl(model, tokenizer, dataset_text, bits=args.bits, use_tq=True)
    print(f"TurboQuant {args.bits}b PPL: {ppl_tq:.4f}")
    
    diff = ((ppl_tq - ppl_base) / ppl_base) * 100
    print(f"\nDelta PPL: {diff:+.2f}%")
    print(f"Status: {'EXCELLENT' if abs(diff) < 1.5 else 'PASSED' if abs(diff) < 5.0 else 'CHECK QUALITY'}")
    print("------------------------")

if __name__ == "__main__":
    main()
