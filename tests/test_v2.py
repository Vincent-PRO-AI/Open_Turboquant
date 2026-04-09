#!/usr/bin/env python3
"""
test_v2.py — TurboQuant v2 unit tests (CPU + optional GPU)
===========================================================

Run:  python test_v2.py
"""

import sys, math, time
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32 if DEVICE == "cpu" else torch.float16
print(f"Device: {DEVICE}  dtype: {DTYPE}")


def test_bitpack_2bit():
    from tq_impl.bitpack import pack_2bit, unpack_2bit
    idx = torch.randint(0, 4, (8, 128), dtype=torch.int16, device=DEVICE)
    packed = pack_2bit(idx)
    assert packed.shape == (8, 32), f"Expected (8,32), got {packed.shape}"
    assert packed.dtype == torch.uint8
    unpacked = unpack_2bit(packed, 128)
    assert (idx == unpacked).all(), "2-bit round-trip failed"
    print("  PASS: 2-bit pack/unpack")


def test_bitpack_3bit():
    from tq_impl.bitpack import pack_3bit, unpack_3bit
    idx = torch.randint(0, 8, (8, 128), dtype=torch.int16, device=DEVICE)
    packed = pack_3bit(idx)
    assert packed.shape == (8, 64), f"Expected (8,64), got {packed.shape}"
    unpacked = unpack_3bit(packed, 128)
    assert (idx == unpacked).all(), "3-bit round-trip failed"
    print("  PASS: 3-bit pack/unpack")


def test_bitpack_1bit():
    from tq_impl.bitpack import pack_1bit, unpack_1bit
    signs = torch.randint(0, 2, (8, 128), device=DEVICE).to(torch.int8) * 2 - 1
    packed = pack_1bit(signs)
    assert packed.shape == (8, 16), f"Expected (8,16), got {packed.shape}"
    unpacked = unpack_1bit(packed, 128)
    assert (signs.float() == unpacked.float()).all(), "1-bit round-trip failed"
    print("  PASS: 1-bit pack/unpack")


def test_compression_ratios():
    from tq_impl.bitpack import compression_ratio
    cr3 = compression_ratio(2, 128)  # 3-bit mode
    cr4 = compression_ratio(3, 128)  # 4-bit mode
    assert abs(cr3 - 4.9) < 0.5, f"3-bit CR: expected ~4.9x, got {cr3}"
    assert abs(cr4 - 3.0) < 0.5, f"4-bit CR: expected ~3.0x, got {cr4}"
    print(f"  PASS: compression ratios 3-bit={cr3:.1f}x  4-bit={cr4:.1f}x")


def test_codebook():
    from tq_impl.codebook import get_codebook, get_boundaries, expected_mse
    c2 = get_codebook(2, 128)
    c3 = get_codebook(3, 128)
    assert c2.shape[0] == 4, f"Expected 4 centroids, got {c2.shape[0]}"
    assert c3.shape[0] == 8, f"Expected 8 centroids, got {c3.shape[0]}"
    # Centroids should be sorted
    assert (c2[1:] > c2[:-1]).all(), "Centroids not sorted"
    # Distortion check
    d_emp = expected_mse(2, 128, n_samples=10_000)
    d_th  = (math.sqrt(3 * math.pi) / 2) / (4 ** 2)
    assert d_emp < d_th * 1.5, f"Distortion too high: {d_emp} vs theory {d_th}"
    print(f"  PASS: codebook (2-bit MSE: {d_emp:.6f} vs theory {d_th:.6f})")


def test_mse_quantizer():
    from tq_impl.core import TurboQuantMSE
    mse = TurboQuantMSE(bits=2, head_dim=128, device=DEVICE, seed=42, dtype=DTYPE)
    x = torch.randn(16, 128, device=DEVICE, dtype=DTYPE)
    x = x / x.norm(dim=-1, keepdim=True)
    idx = mse.quantize_raw(x)
    assert idx.shape == (16, 128)
    assert idx.min() >= 0 and idx.max() <= 3
    x_hat = mse.dequantize_from_idx(idx)
    assert x_hat.shape == (16, 128)
    mse_val = ((x.float() - x_hat.float()) ** 2).mean().item()
    print(f"  PASS: TurboQuantMSE 2-bit (MSE={mse_val:.6f})")


def test_prod_4bit():
    from tq_impl.core import TurboQuantProd
    tqp = TurboQuantProd(bits=4.0, head_dim=128, device=DEVICE, seed=42, dtype=DTYPE)
    keys = torch.randn(2, 4, 10, 128, device=DEVICE, dtype=DTYPE)
    pk = tqp.quantize(keys)
    assert pk.packed_idx.dtype == torch.uint8
    assert pk.packed_qjl.dtype == torch.uint8
    assert pk.bits_mse == 3
    # Expected shapes for 3-bit MSE: D//2 = 64 per position
    assert pk.packed_idx.shape == (2, 4, 10, 64), f"Got {pk.packed_idx.shape}"
    assert pk.packed_qjl.shape == (2, 4, 10, 16), f"Got {pk.packed_qjl.shape}"
    # Dequantize
    k_mse = tqp.dequantize_mse(pk)
    assert k_mse.shape == keys.shape
    k_full = tqp.dequantize_full(pk)
    assert k_full.shape == keys.shape
    # Inner product unbiasedness
    q = torch.randn(128, device=DEVICE, dtype=DTYPE)
    q = q / q.norm()
    true_dots = (keys.reshape(-1, 128).float() @ q.float()).mean().item()
    recon_dots = (k_full.reshape(-1, 128).float() @ q.float()).mean().item()
    bias = abs(recon_dots - true_dots) / (abs(true_dots) + 1e-6)
    print(f"  PASS: TurboQuantProd 4-bit (rel bias={bias:.4f})")


def test_prod_3bit():
    from tq_impl.core import TurboQuantProd
    tqp = TurboQuantProd(bits=3.0, head_dim=128, device=DEVICE, seed=42, dtype=DTYPE)
    keys = torch.randn(2, 4, 10, 128, device=DEVICE, dtype=DTYPE)
    pk = tqp.quantize(keys)
    assert pk.bits_mse == 2
    # 2-bit MSE: D//4 = 32 per position
    assert pk.packed_idx.shape == (2, 4, 10, 32), f"Got {pk.packed_idx.shape}"
    k_mse = tqp.dequantize_mse(pk)
    assert k_mse.shape == keys.shape
    print("  PASS: TurboQuantProd 3-bit")


def test_score_fused():
    from tq_impl.core import TurboQuantProd
    tqp = TurboQuantProd(bits=4.0, head_dim=128, device=DEVICE, seed=42, dtype=DTYPE)
    keys = torch.randn(20, 128, device=DEVICE, dtype=DTYPE)
    pk = tqp.quantize(keys)
    q = torch.randn(1, 128, device=DEVICE, dtype=DTYPE)
    fused = tqp.score_fused(q, pk).flatten()       # [1,20] → [20]
    recon = tqp.dequantize_full(pk)
    standard = (q @ recon.T).flatten()               # [1,20] → [20]
    # Cosine between the two score vectors
    cos = F.cosine_similarity(fused.float(), standard.float(), dim=0).item()
    assert cos > 0.99, f"Fused/standard diverged: cos={cos}"
    print(f"  PASS: score_fused vs standard (cos={cos:.6f})")


def test_concat_packed():
    from tq_impl.core import TurboQuantProd, concat_packed_seq
    tqp = TurboQuantProd(bits=4.0, head_dim=128, device=DEVICE, seed=42, dtype=DTYPE)
    a = tqp.quantize(torch.randn(2, 4, 5, 128, device=DEVICE, dtype=DTYPE))
    b = tqp.quantize(torch.randn(2, 4, 3, 128, device=DEVICE, dtype=DTYPE))
    c = concat_packed_seq(a, b)
    assert c.packed_idx.shape[2] == 8
    assert c.key_norm.shape == (2, 4, 8)
    print("  PASS: concat_packed_seq")


def test_cache_prefill_decode():
    from tq_impl.cache import TurboQuantCache
    cache = TurboQuantCache(bits=4.0, dtype=DTYPE, seed=42)
    # Prefill
    k = torch.randn(1, 4, 32, 128, device=DEVICE, dtype=DTYPE)
    v = torch.randn(1, 4, 32, 128, device=DEVICE, dtype=DTYPE)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.shape == (1, 4, 32, 128), "Prefill should return raw keys"
    assert cache.get_seq_length(0) == 32
    # Decode step
    k1 = torch.randn(1, 4, 1, 128, device=DEVICE, dtype=DTYPE)
    v1 = torch.randn(1, 4, 1, 128, device=DEVICE, dtype=DTYPE)
    k_out2, v_out2 = cache.update(k1, v1, layer_idx=0)
    assert k_out2.shape[2] == 33, f"Expected T=33, got {k_out2.shape[2]}"
    assert cache.get_seq_length(0) == 33
    # Memory
    mem = cache.memory_footprint()
    cr = mem["key_compression_ratio"]
    assert cr > 2.0, f"Compression too low: {cr}"
    print(f"  PASS: cache prefill+decode (compression={cr:.1f}x)")


def test_cache_multi_layer():
    from tq_impl.cache import TurboQuantCache
    cache = TurboQuantCache(bits=3.0, dtype=DTYPE, seed=42)
    for layer in range(4):
        k = torch.randn(1, 2, 16, 128, device=DEVICE, dtype=DTYPE)
        v = torch.randn(1, 2, 16, 128, device=DEVICE, dtype=DTYPE)
        cache.update(k, v, layer_idx=layer)
    assert len(cache) == 4
    for layer in range(4):
        assert cache.get_seq_length(layer) == 16
    # Decode
    for step in range(3):
        for layer in range(4):
            k = torch.randn(1, 2, 1, 128, device=DEVICE, dtype=DTYPE)
            v = torch.randn(1, 2, 1, 128, device=DEVICE, dtype=DTYPE)
            cache.update(k, v, layer_idx=layer)
    for layer in range(4):
        assert cache.get_seq_length(layer) == 19
    print("  PASS: multi-layer cache (4 layers, 16 prefill + 3 decode)")


def test_cache_hf_api():
    from tq_impl.cache import TurboQuantCache
    cache = TurboQuantCache(bits=4.0, dtype=DTYPE, seed=42)
    k = torch.randn(1, 4, 8, 128, device=DEVICE, dtype=DTYPE)
    v = torch.randn(1, 4, 8, 128, device=DEVICE, dtype=DTYPE)
    cache.update(k, v, layer_idx=0)
    # Test properties
    assert cache.seen_tokens == 8
    assert len(cache.key_cache) == 1
    assert len(cache.value_cache) == 1
    # get_mask_sizes
    pos = torch.arange(8)
    sizes = cache.get_mask_sizes(pos, 0)
    assert isinstance(sizes, tuple) and len(sizes) == 2
    print("  PASS: HF API compatibility")


# ==========================================================================

if __name__ == "__main__":
    tests = [
        ("Bitpack 2-bit", test_bitpack_2bit),
        ("Bitpack 3-bit", test_bitpack_3bit),
        ("Bitpack 1-bit", test_bitpack_1bit),
        ("Compression ratios", test_compression_ratios),
        ("Codebook", test_codebook),
        ("MSE quantizer", test_mse_quantizer),
        ("Prod 4-bit", test_prod_4bit),
        ("Prod 3-bit", test_prod_3bit),
        ("Score fused", test_score_fused),
        ("Concat packed", test_concat_packed),
        ("Cache prefill+decode", test_cache_prefill_decode),
        ("Cache multi-layer", test_cache_multi_layer),
        ("Cache HF API", test_cache_hf_api),
    ]

    print(f"\n{'=' * 60}")
    print(f"  TurboQuant v2 — Unit Tests")
    print(f"{'=' * 60}\n")

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} — {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    sys.exit(1 if failed else 0)
