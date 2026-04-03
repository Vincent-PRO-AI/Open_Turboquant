"""
tq_impl/codebook.py
-------------------
Lloyd-Max optimal codebooks for TurboQuant_mse.

After a random rotation, each coordinate of a d-dimensional unit-norm vector
follows approximately N(0, 1/d) by concentration-of-measure.

We pre-compute the Lloyd-Max quantizer centroids for this distribution and
cache them on disk so that subsequent runs are instantaneous.

References
----------
    Paper §3.1 (Algorithm 1) — QUANT_mse constructs codebook by minimising
    the MSE cost in Eq. (4) via solving a 1-D k-means problem.
"""
from __future__ import annotations

import os
import pickle
from functools import lru_cache
from typing import Dict

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Lloyd-Max solver
# ---------------------------------------------------------------------------

def _lloyd_max(n_levels: int, sigma: float, n_iter: int = 2000) -> np.ndarray:
    """
    Compute optimal Lloyd-Max quantizer for N(0, sigma²).

    Alternates between:
      1. Centroid update  : c_i = E[X | X in Voronoi(c_i)]
      2. Boundary update  : b_i = (c_i + c_{i+1}) / 2

    Convergence is tight (< 1e-12 centroid shift) within ~200 iterations.
    """
    from scipy.stats import norm as sp_norm

    # Initialise centroids at quantile midpoints — better than uniform init
    probs = np.linspace(1.0 / (2 * n_levels), 1.0 - 1.0 / (2 * n_levels), n_levels)
    levels = sigma * sp_norm.ppf(probs)

    for _ in range(n_iter):
        prev = levels.copy()

        # Voronoi boundaries  (−∞, b_1, …, b_{n-1}, +∞)
        boundaries = np.concatenate([[-np.inf], (levels[:-1] + levels[1:]) / 2, [np.inf]])

        for i in range(n_levels):
            lo_s = boundaries[i] / sigma
            hi_s = boundaries[i + 1] / sigma

            # P(lo < X < hi) for X ~ N(0, sigma²)
            cdf_lo = sp_norm.cdf(lo_s)
            cdf_hi = sp_norm.cdf(hi_s)
            p = cdf_hi - cdf_lo
            if p < 1e-15:
                continue

            # E[X | lo < X < hi] = sigma * (φ(lo_s) − φ(hi_s)) / p
            pdf_lo = sp_norm.pdf(lo_s)
            pdf_hi = sp_norm.pdf(hi_s)
            levels[i] = sigma * (pdf_lo - pdf_hi) / p

        if np.max(np.abs(levels - prev)) < 1e-12:
            break

    return levels


# ---------------------------------------------------------------------------
# Codebook cache (disk + memory)
# ---------------------------------------------------------------------------

_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".codebook_cache")


def _cache_path(bits: int, head_dim: int) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"b{bits}_d{head_dim}.pkl")


@lru_cache(maxsize=128)
def get_codebook(bits: int, head_dim: int) -> torch.Tensor:
    """
    Return Lloyd-Max centroids tensor of shape [2^bits] (on CPU, float32).

    The distribution is N(0, 1/head_dim), matching the coordinate distribution
    of unit-norm vectors after a random rotation in R^head_dim.

    Results are cached to disk after first computation.
    """
    path = _cache_path(bits, head_dim)
    if os.path.exists(path):
        with open(path, "rb") as f:
            centroids = pickle.load(f)
        return torch.tensor(centroids, dtype=torch.float32)

    sigma = 1.0 / (head_dim ** 0.5)
    n_levels = 2 ** bits
    centroids = _lloyd_max(n_levels, sigma)
    with open(path, "wb") as f:
        pickle.dump(centroids, f)
    return torch.tensor(centroids, dtype=torch.float32)


@lru_cache(maxsize=128)
def get_boundaries(bits: int, head_dim: int) -> torch.Tensor:
    """
    Return (2^bits − 1) Voronoi decision boundaries for torch.bucketize.
    """
    c = get_codebook(bits, head_dim)
    return (c[:-1] + c[1:]) / 2


# ---------------------------------------------------------------------------
# Quick distortion sanity-check (used in benchmark)
# ---------------------------------------------------------------------------

def expected_mse(bits: int, head_dim: int, n_samples: int = 100_000) -> float:
    """
    Empirically estimate D_mse for this codebook on random unit vectors.

    Should be ≈ (√3π / 2) · (1/4^bits) from Theorem 1.
    """
    c = get_codebook(bits, head_dim)
    boundaries = get_boundaries(bits, head_dim)

    x = torch.randn(n_samples, head_dim)
    x = x / x.norm(dim=-1, keepdim=True)  # unit sphere

    # Random rotation
    Pi, _ = torch.linalg.qr(torch.randn(head_dim, head_dim))
    y = x @ Pi.T  # rotated coordinates ~ N(0, 1/d)

    idx = torch.bucketize(y, boundaries)
    y_hat = c[idx]
    x_hat = y_hat @ Pi  # inverse rotation

    mse = ((x - x_hat) ** 2).mean().item()
    return mse
