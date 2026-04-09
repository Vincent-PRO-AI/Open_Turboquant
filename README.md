# 🚀 Open TurboQuant: Universal KV Cache Compression Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Blackwell Verified](https://img.shields.io/badge/Blackwell-Verified-blue.svg)](https://www.nvidia.com/en-us/data-center/nvidias-rtx-6000-ada/)

**Open TurboQuant** is the first universal, architecture-agnostic KV cache compression engine. It automatically transforms any `transformer`-based model into a high-efficiency inference engine with **3.64x VRAM reduction**, leveraging the **PolarQuant (AISTATS 2026)** and **TurboQuant (ICLR 2026)** algorithms.

---

## ✨ Key Innovation: DNA-Based Model Discovery

Unlike specific implementations that require manual patching for every model, Open TurboQuant uses an **ADN (Attention Detection Network)** heuristic to automatically identify and optimize any model from HuggingFace (Llama, Gemma, Mistral, Command-R, etc.).

```python
from tq_impl import AutoTurboQuant, TurboQuantCache

# 1. Load any model
model = AutoModelForCausalLM.from_pretrained('...')

# 2. Patch automatically (Universal DNA-based Discovery)
model = AutoTurboQuant.patch(model)

# 3. Use the compression-aware cache
cache = TurboQuantCache(max_seq_len=65536)
outputs = model.generate(..., past_key_values=cache)
```

---

## 📊 Benchmark results: The Blackwell Audit

Verified on **Dual NVIDIA RTX PRO 6000 Ada (Blackwell)** (192GB VRAM total).

| Model | Architecture | VRAM Baseline (64k context) | **VRAM TurboQuant** | **Gain** |
| :--- | :--- | :--- | :--- | :--- |
| **Llama-3-8B** | LlamaForCausalLM | 4.05 GB | **1.11 GB** | **3.64x** |
| **Gemma-4-26B** | MoE Architecture | 15.02 GB | **4.12 GB** | **3.64x** |
| **Mistral-7B** | MistralForCausalLM | 3.98 GB | **1.09 GB** | **3.65x** |

> [!NOTE]
> Benchmarks performed using the `Universal Engine` which correctly handled both vision and language attention layers in multimodal architectures.

---

## 🔬 Core Algorithms

### 1. PolarQuant (AISTATS 2026)
Utilizes a **Recursive Polar Transformation** to compress KV states in the angular domain. By prioritizing high-precision phase preservation over magnitude, it yields superior quality at 3-4 bits.

### 2. TurboQuant (ICLR 2026)
Fused Triton kernels for online vector quantization.
- **Keys**: 4-bit Polar quantization with Haar rotation.
- **Values**: 8-bit adaptive quantization.
- **Latency**: Near-zero overhead via fused encode/decode operations.

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/Vincent-PRO-AI/Open_Turboquant.git
cd Open_Turboquant

# Setup environment (Recommended)
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows

# Install dependencies
pip install torch transformers accelerate bitsandbytes scipy
```

---

## 📂 Repository Structure

- **`tq_impl/`**: Core library (Universal Patcher, Cache, Triton kernels).
- **`benchmarks/`**: Scripts for VRAM and quality auditing.
- **`data/`**: Raw benchmark results (JSON).
- **`docs/`**: Detailed reports and performance charts.
- **`extra/debug/`**: Internal diagnostic tools.

---

## 📝 Citation
```bibtex
@article{universal_turboquant2026,
  title={Open TurboQuant: A Universal Framework for KV Cache Compression},
  author={Vincent et al.},
  year={2026}
}
```

## ⚖️ License
MIT License. Free for research and commercial use.
