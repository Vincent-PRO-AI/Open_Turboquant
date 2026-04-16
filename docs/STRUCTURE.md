# Repository Structure (Production-Ready)

## Core Library (push to GitHub)

```
turboquant/
├── tq_impl/                      # Main package
│   ├── __init__.py              # Package exports
│   ├── core.py                  # TurboQuantMSE, TurboQuantProd (Algo 1&2)
│   ├── cache.py                 # TurboQuantCache (HF-compatible, 400+ lines)
│   ├── bitpack.py               # Bit-packing utilities (2/3/4/1-bit)
│   ├── codebook.py              # Lloyd-Max codebooks + angular codebooks
│   ├── polar.py                 # Recursive polar transform
│   ├── polar_quant.py           # Hierarchical angle quantization
│   ├── triton_polar.py          # Fused Triton kernels for encode/decode
│   ├── value_quant.py           # Value quantization (FP8/INT8/INT4)
│   └── model_patch.py           # HuggingFace model patching
│
├── demo_turboquant.py           # Simple demo script
├── comprehensive_benchmark.py   # Full benchmark suite
├── test_v2.py                   # 13 unit tests (MUST PASS)
├── setup.py                     # Package metadata + installation
├── requirements.txt             # Dependencies
├── README.md                    # Production documentation
├── .gitignore                   # Git ignore rules
└── LICENSE                      # MIT License

```

## What to Push

### Essential
- `tq_impl/` (all 11 modules)
- `test_v2.py` (proof of correctness)
- `demo_turboquant.py` (entry point)
- `comprehensive_benchmark.py` (reproducibility)
- `requirements.txt` (dependencies)
- `setup.py` (installation)
- `README.md` (documentation)
- `.gitignore` (cleanup)

### Optional but nice
- `vram_stress.py` (GPU stress testing)
- License file (MIT)
- CHANGELOG.md (version history)

## What NOT to Push (use .gitignore)

- `diag_*.py` (15 diagnostic scripts)
- `test_*.py` (except test_v2.py)
- `playground.py`, `run_*.py` (variants)
- `inspect_*.py`, `check_*.py` (inspection tools)
- `__pycache__/`, `*.pyc`, `*.egg-info/`
- Model weights (`*.bin`, `*.pt`)
- Logs and cache files

## Installation for Users

```bash
# From GitHub
git clone https://github.com/vincentsoule/turboquant
cd turboquant
pip install -e .

# Or with Triton
pip install -e ".[triton]"

# Verify
python test_v2.py -v
```

## File Sizes (prod-ready)

| File | Lines | Purpose |
|------|-------|---------|
| cache.py | 410 | Core cache implementation |
| triton_polar.py | 280 | GPU kernels |
| core.py | 180 | Quantization algorithms |
| model_patch.py | 300 | HF integration |
| total | ~2000 | Entire library |

