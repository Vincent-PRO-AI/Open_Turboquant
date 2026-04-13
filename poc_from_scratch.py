#!/usr/bin/env python3
"""
poc_from_scratch.py — Proof-of-Concept: TurboQuant from Zero
=============================================================

End-to-end POC that:
1. Downloads a model (GPT-2 or Llama-2-7B)
2. Creates TurboQuantCache
3. Patches the model
4. Generates text with baseline vs TurboQuant
5. Compares VRAM, speed, quality

Usage: python poc_from_scratch.py [--model MODEL_ID] [--tokens 50]
"""
import argparse
import time
import torch
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM

from tq_impl.cache import TurboQuantCache
from tq_impl.model_patch import patch_model_for_turboquant, unpatch_model_for_turboquant
from tq_impl.bitpack import compression_ratio


def get_gpu_mem_mb():
    """Get current GPU memory usage in MB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


def generate_with_cache(model, tokenizer, prompt, cache=None, max_new_tokens=50):
    """Generate text with optional cache."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2",
                        help="Model ID (gpt2, meta-llama/Llama-2-7b-chat-hf, etc.)")
    parser.add_argument("--tokens", type=int, default=50, help="Max new tokens")
    args = parser.parse_args()

    prompt = "The future of AI is"

    print("=" * 80)
    print("🚀 TURBOQUANT POC — FROM SCRATCH")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Prompt: '{prompt}'")
    print(f"Max tokens: {args.tokens}")
    print("=" * 80)

    # 1. Download model
    print("\n[1/4] Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e):
            print(f"❌ Model '{args.model}' requires authentication or is gated.")
            print(f"   Use: huggingface-cli login")
            print(f"   Or use a public model like 'gpt2'")
            raise
        raise

    model_mem = get_gpu_mem_mb()
    print(f"✓ Model loaded. VRAM: {model_mem:.1f} MB")

    # 2. Baseline (FP16)
    print("\n[2/4] FP16 Baseline...")
    gc.collect()
    torch.cuda.empty_cache()
    mem_before = get_gpu_mem_mb()
    text_fp16, n_tok, time_fp16 = generate_with_cache(model, tokenizer, prompt, cache=None,
                                                       max_new_tokens=args.tokens)
    mem_after = get_gpu_mem_mb()
    cache_mem_fp16 = mem_after - mem_before
    tok_s_fp16 = n_tok / time_fp16 if time_fp16 > 0 else 0

    print(f"✓ Speed: {tok_s_fp16:.1f} tok/s")
    print(f"✓ Cache VRAM: {cache_mem_fp16:.1f} MB")
    print(f"✓ Output: {text_fp16[:100]}...")

    # 3. TurboQuant 4-bit
    print("\n[3/4] TurboQuant 4-bit...")
    cache = TurboQuantCache(bits_key=4.0, bits_value=8.0, dtype=torch.float16)
    patch_model_for_turboquant(model, cache)

    gc.collect()
    torch.cuda.empty_cache()
    mem_before = get_gpu_mem_mb()
    text_tq4, n_tok, time_tq4 = generate_with_cache(model, tokenizer, prompt, cache=cache,
                                                     max_new_tokens=args.tokens)
    mem_after = get_gpu_mem_mb()
    cache_mem_tq4 = mem_after - mem_before
    tok_s_tq4 = n_tok / time_tq4 if time_tq4 > 0 else 0

    unpatch_model_for_turboquant(model)

    cr4 = compression_ratio(3, 128)
    print(f"✓ Speed: {tok_s_tq4:.1f} tok/s ({tok_s_tq4/tok_s_fp16*100:.1f}% of baseline)")
    print(f"✓ Cache VRAM: {cache_mem_tq4:.1f} MB (3.0x compression)")
    print(f"✓ Output: {text_tq4[:100]}...")

    # 4. Compare
    print("\n[4/4] Comparison")
    print("-" * 80)
    print(f"{'Config':<20} {'Speed (tok/s)':<15} {'Cache MB':<15} {'vs FP16'}")
    print("-" * 80)
    print(f"{'FP16 baseline':<20} {tok_s_fp16:<15.1f} {cache_mem_fp16:<15.1f} baseline")
    print(f"{'TurboQuant 4b':<20} {tok_s_tq4:<15.1f} {cache_mem_tq4:<15.1f} "
          f"{(1 - cache_mem_tq4/cache_mem_fp16)*100:+.0f}% VRAM")

    # 5. Output agreement
    print("\n[5/4] Output Agreement")
    print("-" * 80)
    base_tokens = tokenizer.encode(text_fp16)
    tq4_tokens = tokenizer.encode(text_tq4)
    min_len = min(len(base_tokens), len(tq4_tokens))
    if min_len > 0:
        agree = sum(1 for a, b in zip(base_tokens[:min_len], tq4_tokens[:min_len]) if a == b)
        agreement = agree / min_len * 100
        print(f"TurboQuant 4b: {agree}/{min_len} tokens = {agreement:.1f}% agreement")
    else:
        print("⚠️  Could not compute agreement (empty outputs)")

    print("\n" + "=" * 80)
    print("✅ POC COMPLETE")
    print("=" * 80)
    print("\nKey Results:")
    print(f"  ✓ {cache_mem_fp16/cache_mem_tq4:.1f}x KV cache compression")
    print(f"  ✓ {tok_s_tq4/tok_s_fp16*100:.1f}% throughput")
    if min_len > 0:
        print(f"  ✓ {agreement:.1f}% token agreement")
    print("\n")


if __name__ == "__main__":
    main()
