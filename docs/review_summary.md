# TurboQuant V2 — Technical Review Summary for Claude Opus

This document provides a concentrated overview of the **TurboQuant V2** implementation, intended for an expert-level technical review.

## 1. Core Architecture

The project implements **Near-Optimal KV Cache Compression** through a hybrid quantization scheme:
*   **MSE-Optimal Scalar Quantization**: For the bulk of the key vector coordinates (2-bit or 3-bit).
*   **Quantized Johnson-Lindenstrauss (QJL)**: A 1-bit residual correction that ensures unbiased inner products and near-optimal distortion.
*   **Outlier Retention**: Dynamic preservation of critical activations (top 6.25%) in FP16 to ensure 100% Top-1 agreement with the baseline.

## 2. Key Modules

### `tq_impl/cache.py` (The Heart)
- **`TurboQuantCache`**: Subclass of `DynamicCache` (with `transformers` 4.45+ compatibility).
- **Storage**: Uses `uint8` tensors for bit-packed indices (`_packed_keys`) and FP16 for values (`_values`) and outliers (`_outlier_vals`).
- **Prefill vs Decode**: Prefill stores raw FP16 keys in `_raw_keys` for maximum accuracy during the initial prompt. Compression is triggered during the first decode step via `_compress_layer`.

### `tq_impl/core.py` & `tq_impl/codebook_cache/`
- Implements the Optimal Scalar Quantizer using Lloyd-Max algorithm for a Gaussian distribution.
- Pre-calculates centroids for fast lookup.

### `tq_impl/triton_kernel.py`
- Fused Triton kernel for attention scoring directly on bit-packed keys.
- **Scoring Formula**: `score = ||k|| * ||q|| * (<Pi*q_hat, centroid[idx]> + (scale) * <S*q_hat, sign_bit>)`.
- **Optimization**: Extracts 2/3-bit indices and 1-bit signs using bitwise shifts and masks within the GPU kernel to avoid full decompression to VRAM.

### `tq_impl/model_patch.py`
- Extensive monkey-patching suite.
- **Specialty**: Supports `Gemma4TextAttention` and standard `LlamaAttention` architectures.
- **Correctness**: Handles complex `past_key_values` (plural) vs `past_key_value` signatures and architecture-specific norms (`q_norm`, `k_norm`).

## 3. Points for Critical Review

1.  **RoPE Order in Fused Path**: Verification that `apply_rotary_pos_emb` is correctly applied to `q` and `k` *after* projection norms but *before* the fused scoring logic.
2.  **Outlier Scattering**: In `TurboQuantCache._add_outliers`, check the robustness of the `scatter_` operation for multi-head GQA (Grouped Query Attention) where head dimensions might be interleaved.
3.  **Triton Bit-unpacker**: In `TurboQuant_prod_kernel`, verify that the bit-offset logic for 3-bit indices (not power-of-two) doesn't cause alignment issues across blocks.
4.  **Scaling factors**: Ensure the normalization factors (e.g., `sqrt(pi/2)/d`) in the QJL correction are numerically stable for different head dimensions (e.g., 128 vs 96).

## 4. Current Test Results
- **Quality**: 100% Top-1 agreement on Gemma-4-E2B and Llama-3-8B.
- **Compression**: Up to 4.9x (3-bit mode) for Key Cache.
- **Connectivity**: Fully compatible with `model.generate(past_key_values=cache)`.

---
*Summary prepared by Antigravity AI for Vincent's TurboQuant Project.*
