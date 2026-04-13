#!/usr/bin/env python3
"""
setup_validation.py — Validate TurboQuant installation & imports
=================================================================

Tests that the package is correctly installed and all imports work.
Run this first to verify the development environment.

Usage: python setup_validation.py
"""
import sys
import subprocess

print("-" * 80)
print("TURBOQUANT INSTALLATION VALIDATION")
print("-" * 80)

# Step 1: Check Python version
print("\n[1/5] Python Version")
print("-" * 80)
py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print(f"OK: Python {py_version}")
if sys.version_info < (3, 9):
    print("❌ ERROR: Python 3.9+ required")
    sys.exit(1)

# Step 2: Check dependencies
print("\n[2/5] Core Dependencies")
print("-" * 80)
dependencies = {
    "torch": "PyTorch (tensor ops)",
    "transformers": "HuggingFace (models)",
    "numpy": "NumPy (arrays)",
}

all_ok = True
for pkg, desc in dependencies.items():
    try:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "unknown")
        print(f"OK: {pkg:<15} v{version:<20} ({desc})")
    except ImportError:
        print(f"ERROR: {pkg:<15} NOT INSTALLED")
        all_ok = False

if not all_ok:
    print("\nERROR: Missing dependencies. Run: pip install -r requirements.txt")
    sys.exit(1)

# Step 3: Check Triton (optional)
print("\n[3/5] Optional Dependencies")
print("-" * 80)
try:
    import triton
    print(f"OK: triton v{triton.__version__:<20} (GPU kernels - optional)")
except ImportError:
    print("WARN: triton NOT INSTALLED (falls back to PyTorch kernels)")

# Step 4: Test imports
print("\n[4/5] Package Imports")
print("-" * 80)
sys.path.insert(0, ".")

try:
    from tq_impl import (
        TurboQuantMSE, TurboQuantProd, PackedKeys,
        TurboQuantCache,
        patch_model_for_turboquant, unpatch_model_for_turboquant,
        get_codebook, get_boundaries, expected_mse,
        compression_ratio, packed_bytes_per_position,
        recursive_polar_transform, recursive_polar_inverse,
        PolarAngleQuantizer,
        ValueQuantizer,
        is_triton_available, triton_version,
    )
    print("OK: tq_impl.core exports")
    print("  - TurboQuantMSE, TurboQuantProd, PackedKeys")
    print("OK: tq_impl.cache exports")
    print("  - TurboQuantCache, patch/unpatch_model_for_turboquant")
    print("OK: tq_impl utilities")
    print("  - codebook, bitpack, polar, value_quant, triton_polar")
    print(f"OK: Triton available: {is_triton_available()}")
    print(f"OK: Triton version: {triton_version()}")
except Exception as e:
    print(f"ERROR: Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Run unit tests
print("\n[5/5] Unit Tests")
print("-" * 80)
print("Running test_v2.py (13 tests)...\n")

result = subprocess.run([sys.executable, "tests/test_v2.py"], cwd=".")
if result.returncode != 0:
    print("\nERROR: Tests failed")
    sys.exit(1)

print("\n" + "=" * 80)
print("SUCCESS: INSTALLATION VALIDATION PASSED")
print("=" * 80)
print("\nNext steps:")
print("  1. python poc_from_scratch.py    # Run POC with real model")
print("  2. python comprehensive_benchmark.py  # Full performance test")
print("\n")
