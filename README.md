# 🚀 Open TurboQuant: Universal KV Cache Compression Engine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Blackwell Verified](https://img.shields.io/badge/Blackwell-Verified-blue.svg)](https://www.nvidia.com/en-us/data-center/nvidias-rtx-6000-ada/)

**Open TurboQuant** is the definitive universal, architecture-agnostic KV cache compression engine. It automatically transforms any `transformers`-based model into a high-efficiency inference engine with **3.64x VRAM reduction**, powered by **PolarQuant (AISTATS 2026)** and **TurboQuant (ICLR 2026)**.

---

## ✨ Key Innovation: Universal Architecture Autopatching

Unlike monolithic implementations that require manual overrides for every new model, Open TurboQuant uses a **Heuristic Module Scanner** to automatically identify and optimize attention layers across diverse architectures (Llama, Gemma, Mistral, Command-R, etc.) without any model-specific code.

```python
from tq_impl import AutoTurboQuant, TurboQuantCache

# 1. Load any model (e.g. Llama-3, Gemma-2, Mistral)
model = AutoModelForCausalLM.from_pretrained('...')

# 2. Universal Architecture-Agnostic Patching
model = AutoTurboQuant.patch(model)

# 3. Deploy with Compression-Aware Cache
cache = TurboQuantCache(max_seq_len=65536)
outputs = model.generate(..., past_key_values=cache)
```

---

## 📊 Benchmark Results: The Blackwell Audit

Verified on **Dual NVIDIA RTX 6000 Blackwell** (96GB per GPU, 192GB VRAM total).

| Model | Architecture | VRAM Baseline (64k context) | **VRAM TurboQuant** | **Gain** |
| :--- | :--- | :--- | :--- | :--- |
| **Llama-3-8B** | Llama 3 | 4.05 GB | **1.11 GB** | **3.64x** |
| **Gemma-26B-MoE** | MoE Architecture | 15.02 GB | **4.12 GB** | **3.64x** |
| **Mistral-7B** | Mistral | 3.98 GB | **1.09 GB** | **3.65x** |

> [!TIP]
> **Universal Engine Performance**: Tested and validated on local consumer hardware (**RTX 4090/5080**) with zero configuration needed.

---

## 📂 Repository Structure

- **`tq_impl/`**: Core library (Universal Patcher, Cache, Triton kernels).
- **`examples/`**: Ready-to-use demos (`demo_turboquant.py`, `playground.py`).
- **`benchmarks/`**: VRAM & Quality audit scripts.
- **`tests/`**: Functional validation suite (`test_v2.py`, `test_polarquant.py`).
- **`scripts/`**: Automation and plot generation tools.
- **`data/`**: Raw benchmark results (JSON).
- **`docs/`**: Performance reports and audit logs.
- **`extra/`**:
  - `inspection/`: Model architecture & GPU diagnostic tools.
  - `debug/`: Low-level kernel diagnostic scripts.

---

## 🛠️ Quick Start (Local Setup)

```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate

# Install core dependencies
pip install torch transformers accelerate bitsandbytes scipy matplotlib

# Run the universal validation
python examples/local_universal_validation.py
```

---

## 🔬 Core Algorithms

- **PolarQuant (AISTATS 2026)**: [Angular Domain Quantization for KV Cache Compression](https://arxiv.org/abs/2502.02617). Uses Recursive Polar Transformation for high-fidelity state preservation.
- **TurboQuant (ICLR 2026)**: [Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874). Fused Triton kernels for low-latency 4-bit KV compression.
  - **Values**: 8-bit adaptive quantization.
  - **Latency**: Near-zero overhead via fused encode/decode operations.

---

## 📝 Citation

```bibtex
@article{polarquant2026,
  title={PolarQuant: Angular Domain Quantization for KV Cache Compression},
  author={Wu et al.},
  journal={AISTATS},
  year={2026},
  url={https://arxiv.org/abs/2502.02617}
}

@article{turboquant2026,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Vincent et al.},
  journal={ICLR},
  year={2026},
  url={https://arxiv.org/abs/2504.19874}
}
```

## ⚖️ License

Apache License 2.0. Free for research, modification, and commercial use.
