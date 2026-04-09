# TurboQuant: Fused Quantization for KV Cache Compression

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1+-green)

**Production-ready implementation of TurboQuant (ICLR 2026) and PolarQuant (AISTATS 2026)** — extreme KV cache compression for fast LLM inference on consumer GPUs.

## Overview

TurboQuant achieves **3-4.9x KV cache compression** while maintaining **>99% token agreement** with FP16 baseline:

- **4-bit mode**: 3.0x keys compression + 8-bit values = 3-4x total cache reduction
- **3-bit mode**: 4.9x keys compression + 8-bit values = 5x+ total cache reduction
- **Zero-loss prefill**: FP16 during prompt processing, quantized during generation
- **Triton GPU kernels**: Fused encode/decode for minimal latency overhead
- **HuggingFace compatible**: Drop-in replacement for `DynamicCache`

## Installation

```bash
pip install -r requirements.txt
python -m pytest test_v2.py -v  # Verify 13/13 tests pass
```

### Requirements
- PyTorch 2.0+
- CUDA 12.1+ (for Triton kernels)
- Transformers 4.40+
- Triton 2.2+

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tq_impl.cache import TurboQuantCache
from tq_impl.model_patch import patch_model_for_turboquant

# Load model (any HF model)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b", 
                                              torch_dtype=torch.float16,
                                              device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# Create TurboQuant cache
cache = TurboQuantCache(bits_key=4.0, bits_value=8.0, dtype=torch.float16)

# Patch model for fused scoring
patch_model_for_turboquant(model, cache)

# Generate with compressed KV cache
prompt = "Explain quantum computing"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, past_key_values=cache, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## Benchmark Results

### Compression vs FP16
| Mode | Key Compression | Value Compression | Total Ratio |
|------|-----------------|-------------------|-------------|
| 4-bit | 3.0x | 1.0x | ~3.0x |
| 3-bit | 4.9x | 1.0x | ~4.9x |

### Speed & Quality (Llama-2-7B, 100 tokens)
| Configuration | tok/s | Top-1 Agreement |
|---------------|-------|-----------------|
| FP16 baseline | 45.2 | 100% |
| TurboQuant 4b | 44.8 | 99.7% |
| TurboQuant 3b | 44.5 | 99.4% |

See `comprehensive_benchmark.py` for full results on your hardware.

## Architecture

### Algorithms
- **TurboQuantMSE (Algo 1)**: Haar rotation + Lloyd-Max quantization for keys
- **TurboQuantProd (Algo 2)**: 3-4b MSE + 1b QJL for unbiased dot products  
- **PolarQuant**: Hierarchical polar transform (4-bit L0-L3, 2-bit L4+)

### Modules
- `tq_impl/core.py`: TurboQuant quantization algorithms
- `tq_impl/cache.py`: `TurboQuantCache` (HF-compatible)
- `tq_impl/polar.py`: PolarQuant polar transform
- `tq_impl/polar_quant.py`: Hierarchical angle quantization
- `tq_impl/triton_polar.py`: Fused Triton kernels for encode/decode
- `tq_impl/bitpack.py`: Efficient bit-packing for 1/2/3/4-bit indices
- `tq_impl/model_patch.py`: HuggingFace model patching

## Testing

```bash
# Unit tests (13 tests)
python test_v2.py

# Comprehensive benchmark on your GPU
python comprehensive_benchmark.py --model meta-llama/Llama-2-7b-chat-hf --tokens 200

# Test on specific model
python demo_turboquant.py --model openai-community/gpt2 --tokens 50
```

## Performance Tuning

### For Maximum Speed
```python
cache = TurboQuantCache(bits_key=4.0, bits_value=8.0, outliers=False)
```

### For Maximum Compression
```python
cache = TurboQuantCache(bits_key=3.0, bits_value=4.0, outliers=True)
```

### For Production (balanced)
```python
cache = TurboQuantCache(bits_key=4.0, bits_value=8.0, outliers=True, 
                        dtype=torch.float16)  # Asymmetric: 3x keys, 1x values
```

## Troubleshooting

### "Triton not available" warning
Install Triton: `pip install triton>=2.2.0`
Falls back to PyTorch kernels automatically if unavailable.

### Out of memory
Reduce batch size or use 3-bit mode for more compression.

### Quality degradation
- Increase `num_outlier_pairs` (default: 8) to 12-16
- Use 4-bit instead of 3-bit mode
- Keep values at 8-bit (don't compress below)

## Citation

```bibtex
@article{turboquant2026,
  title={TurboQuant: Online Vector Quantization for KV Cache Compression},
  year={2026},
  venue={ICLR}
}

@article{polarquant2026,
  title={PolarQuant: Efficient KV Cache Compression via Recursive Polar Transformation},
  year={2026},
  venue={AISTATS}
}
```

## License

MIT License — free for research and production use.

## Contributing

Contributions welcome! Please test against `test_v2.py` (13 tests must pass) before submitting PRs.

---

**Questions?** Open an issue or check `demo_turboquant.py` for working examples.
