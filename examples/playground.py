#!/usr/bin/env python3
"""
playground.py — TurboQuant vs FP16 baseline benchmark
======================================================
Compare generation quality and memory between:
  - FP16 baseline (standard HF DynamicCache)
  - TurboQuant 4-bit (3b MSE + 1b QJL) = 3.0x compression
  - TurboQuant 3-bit (2b MSE + 1b QJL) = 4.9x compression

Usage:  python playground.py [--model MODEL_ID] [--tokens 100]
"""
import argparse
import time
import torch
import gc
import os
import sys

# Fix pour permettre l'import de tq_impl depuis n'importe quel sous-dossier
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from transformers import AutoTokenizer, AutoModelForCausalLM

from tq_impl import TurboQuantCache, AutoTurboQuant, compression_ratio


def get_gpu_mem_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


def generate(model, tokenizer, prompt, cache=None, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # greedy for reproducibility
        use_cache=True,
    )
    if cache is not None:
        kwargs["past_key_values"] = cache

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    return text, n_new, elapsed


def run_baseline(model, tokenizer, prompt, max_new_tokens):
    """Standard FP16 generation (no TurboQuant)."""
    gc.collect(); torch.cuda.empty_cache()
    mem_before = get_gpu_mem_mb()
    text, n_tok, elapsed = generate(model, tokenizer, prompt, cache=None,
                                    max_new_tokens=max_new_tokens)
    mem_after = get_gpu_mem_mb()
    return dict(
        text=text, tokens=n_tok, time=elapsed,
        tok_s=n_tok / elapsed,
        cache_mb=mem_after - mem_before,
        label="FP16 baseline",
    )


def run_turboquant(model, tokenizer, prompt, bits_key, max_new_tokens):
    """TurboQuant compressed generation."""
    gc.collect(); torch.cuda.empty_cache()

    cache = TurboQuantCache(
        bits_key=bits_key,
        bits_value=8.0,
        outliers=True,
        dtype=torch.float16,
    )
    patch_model_for_turboquant(model, cache)

    mem_before = get_gpu_mem_mb()
    text, n_tok, elapsed = generate(model, tokenizer, prompt, cache=cache,
                                    max_new_tokens=max_new_tokens)
    mem_after = get_gpu_mem_mb()

    unpatch_model_for_turboquant(model)

    cr = compression_ratio(int(bits_key) - 1, 128)
    return dict(
        text=text, tokens=n_tok, time=elapsed,
        tok_s=n_tok / elapsed,
        cache_mb=mem_after - mem_before,
        label=f"TurboQuant {bits_key:.0f}-bit (keys {cr:.1f}x)",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-it",
                        help="HuggingFace model ID")
    parser.add_argument("--tokens", type=int, default=100,
                        help="Max new tokens to generate")
    parser.add_argument("--prompt", default=None,
                        help="Custom prompt (default: built-in)")
    args = parser.parse_args()

    prompt = args.prompt or (
        "Explain the key ideas behind KV cache compression in large language models, "
        "including techniques like quantization, eviction policies, and their trade-offs "
        "for inference speed and output quality."
    )

    print(f"{'=' * 70}")
    print(f"  TurboQuant Playground — Perf Benchmark")
    print(f"{'=' * 70}")
    print(f"  Model  : {args.model}")
    print(f"  GPU    : {torch.cuda.get_device_properties(0).name}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Tokens : {args.tokens}")
    print(f"  Prompt : {prompt[:60]}...")
    print(f"{'=' * 70}\n")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Model loaded. VRAM used: {get_gpu_mem_mb():.0f} MB\n")

    # --- Run benchmarks ---
    results = []

    print("[1/3] FP16 baseline...")
    results.append(run_baseline(model, tokenizer, prompt, args.tokens))
    print(f"       {results[-1]['tok_s']:.1f} tok/s, cache ~{results[-1]['cache_mb']:.1f} MB\n")

    print("[2/3] TurboQuant 4-bit keys...")
    results.append(run_turboquant(model, tokenizer, prompt, 4.0, args.tokens))
    print(f"       {results[-1]['tok_s']:.1f} tok/s, cache ~{results[-1]['cache_mb']:.1f} MB\n")

    print("[3/3] TurboQuant 3-bit keys...")
    results.append(run_turboquant(model, tokenizer, prompt, 3.0, args.tokens))
    print(f"       {results[-1]['tok_s']:.1f} tok/s, cache ~{results[-1]['cache_mb']:.1f} MB\n")

    # --- Summary table ---
    baseline = results[0]
    print(f"{'=' * 70}")
    print(f"  {'Config':<35} {'tok/s':>7} {'Cache MB':>10} {'vs FP16':>8}")
    print(f"  {'-'*35} {'-'*7} {'-'*10} {'-'*8}")
    for r in results:
        speedup = r["tok_s"] / baseline["tok_s"] if baseline["tok_s"] > 0 else 0
        savings = (1 - r["cache_mb"] / baseline["cache_mb"]) * 100 if baseline["cache_mb"] > 0 else 0
        print(f"  {r['label']:<35} {r['tok_s']:>7.1f} {r['cache_mb']:>10.1f} {savings:>+7.0f}%")
    print(f"{'=' * 70}\n")

    # --- Output comparison ---
    print("Output comparison (first 200 chars):")
    for r in results:
        out_text = r["text"][len(prompt):].strip()[:200]
        print(f"\n  [{r['label']}]")
        print(f"  {out_text}")

    # --- Top-1 agreement ---
    if len(results) >= 2:
        base_text = results[0]["text"]
        print(f"\n{'=' * 70}")
        print(f"  Top-1 Token Agreement vs FP16 baseline:")
        base_tokens = tokenizer.encode(base_text)
        for r in results[1:]:
            r_tokens = tokenizer.encode(r["text"])
            min_len = min(len(base_tokens), len(r_tokens))
            if min_len > 0:
                agree = sum(1 for a, b in zip(base_tokens[:min_len], r_tokens[:min_len]) if a == b)
                print(f"  {r['label']:<35} {agree}/{min_len} = {agree/min_len*100:.1f}%")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
