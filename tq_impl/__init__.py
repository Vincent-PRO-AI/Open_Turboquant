"""
tq_impl — TurboQuant KV Cache Compression (ICLR 2026)
======================================================

Open-source implementation of:
    Zandieh et al., "TurboQuant: Online Vector Quantization for KV Cache
    Compression with Near-Optimal Distortion Rate", ICLR 2026

Features
--------
- Bit-packed storage: 4.9x (3-bit) or 3.0x (4-bit) key compression
- Prefill-aware cache: zero quality loss during prefill, compressed decode
- Fused Triton kernels: attention logits without decompressing keys
- Lloyd-Max optimal codebooks with disk caching
- Drop-in HuggingFace generate() compatibility

Usage
-----
    from tq_impl import TurboQuantCache, patch_model_for_turboquant

    cache = TurboQuantCache(bits=4)   # 4-bit: best quality/compression trade-off
    patch_model_for_turboquant(model, cache)
    output = model.generate(input_ids, past_key_values=cache, use_cache=True, ...)
"""

from .core        import TurboQuantMSE, TurboQuantProd, PackedKeys
from .cache       import TurboQuantCache
from .codebook    import get_codebook, get_boundaries, expected_mse
from .model_patch import patch_model_for_turboquant, unpatch_model_for_turboquant
from .triton_kernel import is_triton_available, triton_version
from .bitpack     import compression_ratio, packed_bytes_per_position

__version__ = "2.0.0"
__all__ = [
    "TurboQuantMSE", "TurboQuantProd", "PackedKeys",
    "TurboQuantCache",
    "get_codebook", "get_boundaries", "expected_mse",
    "patch_model_for_turboquant", "unpatch_model_for_turboquant",
    "is_triton_available", "triton_version",
    "compression_ratio", "packed_bytes_per_position",
]
