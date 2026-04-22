#!/usr/bin/env python3
"""
run_benchmark_v3.py — TurboQuant v2 benchmark (bit-packed, prefill-aware)
=========================================================================

Tests both 3-bit (4.9x compression) and 4-bit (3.0x, better quality) modes.
"""

import gc, sys, time, math, os
import torch
import torch.nn.functional as F

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID       = "google/gemma-4-31B-it"
MAX_NEW_TOKENS = 64
CONTEXT_SIZES  = [1024, 8192, 32768]
BIT_MODES      = [4, 3]     # Test 4-bit first (better quality), then 3-bit
TEST_FUSED     = True
TOKEN          = os.getenv("HF_TOKEN")

# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------

print("=" * 78)
print("  TurboQuant v2 Benchmark — bit-packed, prefill-aware")
print("=" * 78)

assert torch.cuda.is_available(), "CUDA required"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}  {p.total_mem / 1024**3:.1f} Go" if hasattr(p, 'total_mem') else f"  GPU {i}: {p.name}  {p.total_memory / 1024**3:.1f} Go")

GPU = "cuda:0"
total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

# ---------------------------------------------------------------------------
# Import tq_impl
# ---------------------------------------------------------------------------

print("\n  Chargement de tq_impl v2...")
from tq_impl import (
    TurboQuantCache,
    patch_model_for_turboquant, unpatch_model_for_turboquant,
    is_triton_available, triton_version,
    expected_mse, compression_ratio,
)

print(f"  Triton: {'v' + triton_version if is_triton_available() else 'non disponible'}")

# Ratios will be displayed after model load to get head_dim
# (The code block was moved below AutoModelForCausalLM.from_pretrained)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

print(f"\n  Chargement {MODEL_ID} (4-bit NF4)...")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    device_map="auto", 
    quantization_config=quantization_config,
    trust_remote_code=True,
    token=TOKEN
)
model.eval()

# Get actual head dim (handle VLMs)
def get_head_dim(cfg):
    if hasattr(cfg, "text_config"): cfg = cfg.text_config
    if hasattr(cfg, "head_dim"): return cfg.head_dim
    return cfg.hidden_size // cfg.num_attention_heads

head_dim = get_head_dim(model.config)
print(f"  Head dimension detectée: {head_dim}")

for b in BIT_MODES:
    cr = compression_ratio(b - 1, head_dim)
    print(f"  {b}-bit mode: {cr:.1f}x compression clés (MSE {b-1}-bit + QJL 1-bit)")

# Codebook sanity
print("\n  Codebooks Lloyd-Max:")
for bits in [2, 3]:
    d_emp = expected_mse(bits, head_dim, n_samples=10_000)
    d_th  = (math.sqrt(3 * math.pi) / 2) / (4 ** bits)
    print(f"    {bits}-bit MSE: D_emp={d_emp:.6f}  D_theorie={d_th:.6f}  {'OK' if d_emp < d_th * 1.5 else 'WARN'}")

model_vram = torch.cuda.memory_allocated(0) / 1024**3
print(f"  Modèle: {model_vram:.2f} Go  |  VRAM libre: {total_vram - model_vram:.2f} Go")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_PROMPT = (
    "Explique en détail la quantification vectorielle pour les modèles de "
    "langage et son application à la compression du cache clé-valeur. "
    "Détaille les compromis entre nombre de bits et qualité. "
)

def build_input(target: int) -> torch.Tensor:
    text = BASE_PROMPT * max(1, target // 35)
    msgs = [
        {"role": "system", "content": "Tu es un assistant expert en ML."},
        {"role": "user",   "content": text},
    ]
    device = next(model.parameters()).device
    try:
        res = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
            max_length=target, truncation=True,
        )
        if isinstance(res, torch.Tensor):
            return res.to(device)
        return res.input_ids.to(device)
    except ValueError:
        # Fallback for models without a chat template (e.g. some base models)
        prompt_text = "Tu es un assistant expert en ML.\nUtilisateur: " + text + "\nAssistant:"
        return tokenizer(
            prompt_text, return_tensors="pt", max_length=target, truncation=True
        ).input_ids.to(device)


def vram_stats():
    return (torch.cuda.memory_allocated(0) / 1024**3,
            torch.cuda.max_memory_allocated(0) / 1024**3)


def run_baseline(ids):
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(0)
    vb, _ = vram_stats()
    try:
        t0 = time.perf_counter()
        with torch.inference_mode():
            # 🚀 Chunked Prefill for Baseline
            CHUNK_SIZE = 1024
            if ids.shape[1] > CHUNK_SIZE:
                # We need a dummy cache to do chunking in native HF
                # No, we can just use the internal cache
                # transformers handles this if we pass use_cache=True
                past = None
                for start in range(0, ids.shape[1] - 1, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, ids.shape[1] - 1)
                    if start >= end: break
                    out_f = model(ids[:, start:end], past_key_values=past, use_cache=True)
                    past = out_f.past_key_values
                out = model.generate(ids[:, -1:], past_key_values=past, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
            else:
                out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache(); return None
    va, vp = vram_stats()
    n = out.shape[1] - ids.shape[1]
    return {"tps": n/dt, "dt": dt, "vram_peak": vp, "kv_delta": va - vb, "n": n}


def run_tq(ids, bits, fused=False):
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(0)
    vb, _ = vram_stats()

    cache = TurboQuantCache(bits=float(bits), dtype=model.dtype)
    if fused:
        patch_model_for_turboquant(model, cache)

    try:
        t0 = time.perf_counter()
        with torch.inference_mode():
            # 🚀 Chunked Prefill for Long Context
            CHUNK_SIZE = 1024
            if ids.shape[1] > CHUNK_SIZE:
                for start in range(0, ids.shape[1] - 1, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, ids.shape[1] - 1)
                    if start >= end: break
                    model(ids[:, start:end], past_key_values=cache, use_cache=True)
                # Last token and generate
                out = model.generate(
                    ids[:, -1:], past_key_values=cache,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
                )
            else:
                out = model.generate(
                    ids, past_key_values=cache,
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
                )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        if fused: unpatch_model_for_turboquant(model)
        return None
    finally:
        if fused: unpatch_model_for_turboquant(model)

    va, vp = vram_stats()
    n = out.shape[1] - ids.shape[1]
    mem = cache.memory_footprint()
    return {"tps": n/dt, "dt": dt, "vram_peak": vp, "kv_delta": va - vb, "n": n, "mem": mem}


# ---------------------------------------------------------------------------
# Quality measurement
# ---------------------------------------------------------------------------

def measure_quality(ids, bits, fused=False):
    n_dec = 8
    with torch.inference_mode():
        # Prefill
        out_b = model(ids, use_cache=True)
        lb = out_b.logits[:, -1, :]

        c = TurboQuantCache(bits=float(bits), dtype=model.dtype)
        if fused:
            patch_model_for_turboquant(model, c)
        try:
            out_t = model(ids, past_key_values=c, use_cache=True)
        finally:
            if fused:
                unpatch_model_for_turboquant(model)
        lt = out_t.logits[:, -1, :]

    cos_pre = F.cosine_similarity(lb, lt).mean().item()
    top1_pre = (lb.argmax(-1) == lt.argmax(-1)).float().mean().item()

    # Decode
    with torch.inference_mode():
        gb = model.generate(ids, max_new_tokens=n_dec, do_sample=False,
                            return_dict_in_generate=True, output_logits=True)
        c2 = TurboQuantCache(bits=float(bits), dtype=model.dtype)
        if fused:
            patch_model_for_turboquant(model, c2)
        try:
            gt = model.generate(ids, past_key_values=c2, max_new_tokens=n_dec,
                                do_sample=False, return_dict_in_generate=True, output_logits=True)
        finally:
            if fused:
                unpatch_model_for_turboquant(model)

    cos_d, top1_d = [], []
    for i in range(min(n_dec, len(gb.logits), len(gt.logits))):
        cos_d.append(F.cosine_similarity(gb.logits[i], gt.logits[i]).mean().item())
        top1_d.append((gb.logits[i].argmax(-1) == gt.logits[i].argmax(-1)).float().mean().item())

    return {
        "cos_pre": cos_pre, "top1_pre": top1_pre,
        "cos_dec": sum(cos_d)/len(cos_d) if cos_d else 0,
        "top1_dec": sum(top1_d)/len(top1_d) if top1_d else 0,
    }


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

print(f"\n{'=' * 78}")
print(f"  BENCHMARK PRINCIPAL")
print(f"{'=' * 78}")

for bits in BIT_MODES:
    cr = compression_ratio(bits - 1, 128)
    print(f"\n  --- {bits}-bit TurboQuant ({cr:.1f}x key compression) ---")
    print(f"  {'Ctx':>8} | {'Mode':<18} | {'tok/s':>7} | {'Temps':>6} | {'VRAM pic':>8} | {'KV delta':>9} | {'Key comp':>9}")
    print(f"  {'-' * 80}")

    for ctx in CONTEXT_SIZES:
        ids = build_input(ctx)
        actual = ids.shape[1]

        # Baseline (only for first bit mode to avoid redundancy)
        if bits == BIT_MODES[0]:
            rb = run_baseline(ids)
            if rb:
                print(f"  {actual:>8} | {'FP16 baseline':<18} | {rb['tps']:>6.1f}t | {rb['dt']:>5.1f}s | {rb['vram_peak']:>6.2f}Go | +{rb['kv_delta']:>7.2f}Go |       —")
            else:
                print(f"  {actual:>8} | {'FP16 baseline':<18} |    OOM |     — |       — |         — |       —")

        # TurboQuant
        rt = run_tq(ids, bits)
        label = f"TQ{bits}b"
        if rt:
            mem = rt.get("mem", {})
            kcr = mem.get("key_compression_ratio", 0)
            print(f"  {actual:>8} | {label:<18} | {rt['tps']:>6.1f}t | {rt['dt']:>5.1f}s | {rt['vram_peak']:>6.2f}Go | +{rt['kv_delta']:>7.2f}Go | {kcr:>7.1f}x")
        else:
            print(f"  {actual:>8} | {label:<18} |    OOM |     — |       — |         — |       —")

        if TEST_FUSED:
            rf = run_tq(ids, bits, fused=True)
            label_f = f"TQ{bits}b fused"
            if rf:
                mem = rf.get("mem", {})
                kcr = mem.get("key_compression_ratio", 0)
                print(f"  {actual:>8} | {label_f:<18} | {rf['tps']:>6.1f}t | {rf['dt']:>5.1f}s | {rf['vram_peak']:>6.2f}Go | +{rf['kv_delta']:>7.2f}Go | {kcr:>7.1f}x")

    print(f"  {'-' * 80}")

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

print(f"\n{'=' * 78}")
print("  QUALITÉ (distorsion des logits)")
print(f"{'=' * 78}")

for bits in BIT_MODES:
    print(f"\n  --- {bits}-bit (standard dequant) ---")
    print(f"  {'Ctx':>8} | {'Prefill cos':>12} | {'Prefill top1':>12} | {'Decode cos':>12} | {'Decode top1':>12}")
    print(f"  {'-' * 65}")
    for ctx in [512, 2048, 4096]:
        try:
            ids = build_input(ctx)
            q = measure_quality(ids, bits, fused=False)
            print(f"  {ids.shape[1]:>8} | {q['cos_pre']:>12.5f} | {q['top1_pre']:>11.1%} | {q['cos_dec']:>12.5f} | {q['top1_dec']:>11.1%}")
        except Exception as e:
            print(f"  {ctx:>8} | erreur: {e}")

if TEST_FUSED:
    for bits in BIT_MODES:
        print(f"\n  --- {bits}-bit (FUSED scoring) ---")
        print(f"  {'Ctx':>8} | {'Prefill cos':>12} | {'Prefill top1':>12} | {'Decode cos':>12} | {'Decode top1':>12}")
        print(f"  {'-' * 65}")
        for ctx in [512, 2048, 4096]:
            try:
                ids = build_input(ctx)
                q = measure_quality(ids, bits, fused=True)
                print(f"  {ids.shape[1]:>8} | {q['cos_pre']:>12.5f} | {q['top1_pre']:>11.1%} | {q['cos_dec']:>12.5f} | {q['top1_dec']:>11.1%}")
            except Exception as e:
                print(f"  {ctx:>8} | erreur: {e}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'=' * 78}")
print("  RÉSUMÉ")
print(f"{'=' * 78}")
print(f"  Modèle        : {MODEL_ID}")
print(f"  GPU           : {torch.cuda.get_device_properties(0).name}")
print(f"  VRAM          : {total_vram:.1f} Go totale, {model_vram:.2f} Go modèle")
print(f"  Triton        : {'v' + triton_version() if is_triton_available() else 'non'}")
for b in BIT_MODES:
    cr = compression_ratio(b - 1, 128)
    print(f"  {b}-bit mode    : {b-1}b MSE + 1b QJL = {cr:.1f}x compression clés")
print(f"{'=' * 78}")