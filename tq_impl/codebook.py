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

# ---------------------------------------------------------------------------
# Lloyd-Max solver
# ---------------------------------------------------------------------------

def _lloyd_max(n_levels: int, sigma: float, n_iter: int = 1000) -> np.ndarray:
    """Optimal Lloyd-Max for N(0, sigma²)."""
    from scipy.stats import norm as sp_norm
    probs = np.linspace(1.0 / (2 * n_levels), 1.0 - 1.0 / (2 * n_levels), n_levels)
    centroids = sigma * sp_norm.ppf(probs)

    for _ in range(n_iter):
        prev = centroids.copy()
        boundaries = np.concatenate([[-np.inf], (centroids[:-1] + centroids[1:]) / 2, [np.inf]])
        for i in range(n_levels):
            lo, hi = boundaries[i] / sigma, boundaries[i + 1] / sigma
            p = sp_norm.cdf(hi) - sp_norm.cdf(lo)
            if p > 1e-15:
                centroids[i] = sigma * (sp_norm.pdf(lo) - sp_norm.pdf(hi)) / p
        if np.max(np.abs(centroids - prev)) < 1e-12: break
    return centroids


def _lloyd_max_angular(n_levels: int, L: int, n_iter: int = 500) -> np.ndarray:
    """
    Optimal Lloyd-Max for f_L(φ) ∝ (sin 2φ)^(2^L - 1) on [0, π/2].
    For L=0, it is uniform on [0, 2π].
    """
    if L == 0:
        # Uniform on [0, 2π]
        return np.linspace(0, 2 * np.pi, n_levels + 1)[:-1] + (np.pi / n_levels)

    # Numerical integration for f_L(φ)
    phi = np.linspace(0, np.pi/2, 2000)
    pdf = (np.sin(2 * phi)) ** (2**L - 1)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]

    # Initial centroids via inverse CDF
    target_cdfs = np.linspace(1.0/(2*n_levels), 1.0 - 1.0/(2*n_levels), n_levels)
    centroids = np.interp(target_cdfs, cdf, phi)

    for _ in range(n_iter):
        prev = centroids.copy()
        bounds = np.concatenate([[0], (centroids[:-1] + centroids[1:]) / 2, [np.pi/2]])
        
        for i in range(n_levels):
            mask = (phi >= bounds[i]) & (phi <= bounds[i+1])
            if np.any(mask):
                centroids[i] = np.average(phi[mask], weights=pdf[mask])
        
        if np.max(np.abs(centroids - prev)) < 1e-10: break
            
    return centroids


# ---------------------------------------------------------------------------
# Codebook cache (disk + memory)
# ---------------------------------------------------------------------------

_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".codebook_cache")

def _path_gaussian(bits: int, head_dim: int) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"gauss_b{bits}_d{head_dim}.pkl")

def _path_angular(bits: int, L: int) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"angle_b{bits}_L{L}.pkl")


@lru_cache(maxsize=128)
def get_codebook(bits: int, head_dim: int) -> torch.Tensor:
    path = _path_gaussian(bits, head_dim)
    if os.path.exists(path):
        with open(path, "rb") as f: return torch.tensor(pickle.load(f), dtype=torch.float32)

    centroids = _lloyd_max(2**bits, 1.0 / (head_dim**0.5))
    with open(path, "wb") as f: pickle.dump(centroids, f)
    return torch.tensor(centroids, dtype=torch.float32)


@lru_cache(maxsize=128)
def get_angular_codebook(bits: int, L: int) -> torch.Tensor:
    path = _path_angular(bits, L)
    if os.path.exists(path):
        with open(path, "rb") as f: return torch.tensor(pickle.load(f), dtype=torch.float32)

    centroids = _lloyd_max_angular(2**bits, L)
    with open(path, "wb") as f: pickle.dump(centroids, f)
    return torch.tensor(centroids, dtype=torch.float32)


def get_boundaries(bits: int, head_dim: int) -> torch.Tensor:
    c = get_codebook(bits, head_dim)
    return (c[:-1] + c[1:]) / 2

def get_angular_boundaries(bits: int, L: int) -> torch.Tensor:
    c = get_angular_codebook(bits, L)
    return (c[:-1] + c[1:]) / 2


def expected_mse(bits: int, head_dim: int, n_samples: int = 10_000) -> float:
    """
    Empirical expected MSE of Lloyd-Max quantizer for N(0, 1/sqrt(d)).
    """
    sigma = 1.0 / (head_dim ** 0.5)
    cb = get_codebook(bits, head_dim)
    bd = get_boundaries(bits, head_dim)

    x = torch.randn(n_samples) * sigma
    idx = torch.bucketize(x, bd)
    x_hat = cb[idx]
    return ((x - x_hat) ** 2).mean().item()


# -------------------------------------------------------------------------