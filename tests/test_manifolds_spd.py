"""Property tests for manifold_helpers.manifolds.spd."""
from __future__ import annotations

import numpy as np

from manifold_helpers.manifolds import spd
from conftest import random_spd


def test_distance_to_self_zero(rng):
    for _ in range(5):
        P = random_spd(rng)
        assert spd.distance(P, P) < 1e-8


def test_distance_symmetric(rng):
    for _ in range(5):
        P = random_spd(rng)
        Q = random_spd(rng)
        assert abs(spd.distance(P, Q) - spd.distance(Q, P)) < 1e-8


def test_exp_log_roundtrip(rng):
    """exp_P(log_P(Q)) = Q for SPD matrices."""
    for _ in range(5):
        P = random_spd(rng)
        Q = random_spd(rng)
        Q_back = spd.exp_map(P, spd.log_map(P, Q))
        assert np.allclose(Q, Q_back, atol=1e-7)


def test_exp_returns_spd(rng):
    """exp_P(V) is SPD: symmetric, positive eigenvalues."""
    for _ in range(5):
        P = random_spd(rng)
        # Random symmetric tangent
        A = rng.normal(scale=0.3, size=P.shape)
        V = (A + A.T) / 2.0
        Q = spd.exp_map(P, V)
        assert np.allclose(Q, Q.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh((Q + Q.T) / 2.0)
        assert np.all(eigvals > 0)


def test_symmetrization_guard_handles_drift(rng):
    """Inputs with ε numerical drift off-symmetric still produce a valid result."""
    P = random_spd(rng)
    # Inject 1e-12 asymmetry
    P_drifted = P + 1e-12 * rng.normal(size=P.shape)
    Q = random_spd(rng)
    d = spd.distance(P_drifted, Q)
    assert np.isfinite(d)
    assert d >= 0
