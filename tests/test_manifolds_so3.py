"""Property tests for ophanimus.manifolds.so3."""
from __future__ import annotations

import numpy as np

from ophanimus.manifolds import so3
from conftest import random_so3_seeded


def test_identity_log_is_zero():
    assert np.allclose(so3.log_map(np.eye(3)), 0.0, atol=1e-12)


def test_exp_log_roundtrip(rng):
    """log(exp(v)) = v for axis-angle vectors with |v| in (0, pi)."""
    for _ in range(10):
        v = rng.uniform(-2.5, 2.5, size=3)
        if np.linalg.norm(v) < 1e-3 or np.linalg.norm(v) > np.pi - 0.05:
            continue
        R = so3.exp_map(v)
        v_back = so3.log_map(R)
        # so3 has antipodal ambiguity at theta=pi; check up to sign in that case
        assert np.allclose(v, v_back, atol=1e-8) or np.allclose(v, -v_back, atol=1e-8)


def test_exp_produces_rotation(rng):
    """exp_map returns a proper rotation (R^T R = I, det R = +1)."""
    for _ in range(10):
        v = rng.uniform(-1.0, 1.0, size=3)
        R = so3.exp_map(v)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


def test_distance_to_self_zero(rng):
    for _ in range(10):
        R = random_so3_seeded(rng)
        assert so3.distance(R, R) < 1e-10


def test_distance_symmetric(rng):
    for _ in range(10):
        R1 = random_so3_seeded(rng)
        R2 = random_so3_seeded(rng)
        assert abs(so3.distance(R1, R2) - so3.distance(R2, R1)) < 1e-10


def test_parallel_transport_norm(rng):
    """SO(3) bi-invariant metric: ||PT v|| = ||v|| (Euclidean norm in so(3))."""
    for _ in range(10):
        R1 = random_so3_seeded(rng)
        R2 = random_so3_seeded(rng)
        w = rng.normal(scale=0.3, size=3)
        w_t = so3.parallel_transport(w, R1, R2)
        assert abs(np.linalg.norm(w) - np.linalg.norm(w_t)) < 1e-10
