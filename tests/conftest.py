"""Shared fixtures for the manifold-helpers test suite.

Most tests use deterministic seeds (np.random.default_rng(0)) so that
optimization-based algorithms (kmeans, regression, embeddings) don't
flake. ATOL is conservative — algorithms with closed forms hit 1e-10,
optimization-based hit 1e-3 or 1e-6 depending on what's being checked.
"""
from __future__ import annotations

import numpy as np
import pytest


ATOL_TIGHT = 1e-8
ATOL_LOOSE = 1e-3


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def atol_tight() -> float:
    return ATOL_TIGHT


@pytest.fixture
def atol_loose() -> float:
    return ATOL_LOOSE


# ----- Sample-point helpers ---------------------------------------------------

def random_sphere_point(rng: np.random.Generator, dim: int = 3) -> np.ndarray:
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


def random_sphere_tangent(rng: np.random.Generator, p: np.ndarray, scale: float = 0.3) -> np.ndarray:
    v = rng.normal(scale=scale, size=p.shape)
    return v - np.dot(p, v) * p  # project off the radial direction


def random_poincare_point(rng: np.random.Generator, dim: int = 3, max_norm: float = 0.7) -> np.ndarray:
    v = rng.normal(size=dim)
    n = np.linalg.norm(v)
    target = rng.uniform(0.05, max_norm)
    return v * (target / n)


def random_hyperboloid_point(rng: np.random.Generator, dim: int = 3) -> np.ndarray:
    """Sample a hyperboloid point via Poincaré -> hyperboloid conversion."""
    from manifold_helpers.manifolds.hyperboloid import from_poincare
    p = random_poincare_point(rng, dim=dim)
    return from_poincare(p)


def random_hyperboloid_tangent(rng: np.random.Generator, p: np.ndarray, scale: float = 0.3) -> np.ndarray:
    """Sample a tangent at p that is Minkowski-orthogonal to p.

    Build any vector v in R^(n+1), then project onto p's tangent space:
      v_tan = v - (<p, v>_M / <p, p>_M) * p,  with <p,p>_M = -1.
    """
    v = rng.normal(scale=scale, size=p.shape)
    inner = -p[0] * v[0] + np.dot(p[1:], v[1:])
    # tangent component: v + inner * p (since <p,p>_M = -1)
    return v + inner * p


def random_so3() -> np.ndarray:
    """Construct a random SO(3) rotation via exp of a random axis-angle vector."""
    from manifold_helpers.manifolds.so3 import exp_map
    rng = np.random.default_rng(1)
    return exp_map(rng.uniform(-np.pi / 2, np.pi / 2, size=3))


def random_so3_seeded(rng: np.random.Generator) -> np.ndarray:
    from manifold_helpers.manifolds.so3 import exp_map
    return exp_map(rng.uniform(-np.pi / 2, np.pi / 2, size=3))


def random_spd(rng: np.random.Generator, n: int = 4) -> np.ndarray:
    """Sample an SPD matrix as A A^T + n*I — well-conditioned by construction."""
    A = rng.normal(size=(n, n))
    return A @ A.T + n * np.eye(n)
