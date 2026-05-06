"""Property tests for ophanimus.manifolds.poincare."""
from __future__ import annotations

import numpy as np

from ophanimus.manifolds import poincare
from conftest import random_poincare_point


def test_distance_to_self_zero(rng):
    for _ in range(10):
        p = random_poincare_point(rng)
        assert poincare.distance(p, p) < 1e-10


def test_distance_symmetric(rng):
    for _ in range(10):
        p = random_poincare_point(rng)
        q = random_poincare_point(rng)
        assert abs(poincare.distance(p, q) - poincare.distance(q, p)) < 1e-10


def test_exp_log_roundtrip_points(rng, atol_tight):
    """exp_p(log_p(q)) = q for two Poincaré points."""
    for _ in range(10):
        p = random_poincare_point(rng)
        q = random_poincare_point(rng)
        q_back = poincare.exp_map(p, poincare.log_map(p, q))
        assert np.allclose(q, q_back, atol=1e-7)


def test_exp_log_roundtrip_tangent(rng, atol_tight):
    """log_p(exp_p(v)) = v for tangent vectors of moderate norm."""
    for _ in range(10):
        p = random_poincare_point(rng, max_norm=0.5)
        v = rng.normal(scale=0.2, size=p.shape)
        q = poincare.exp_map(p, v)
        v_back = poincare.log_map(p, q)
        assert np.allclose(v, v_back, atol=1e-7)


def test_parallel_transport_norm_under_metric(rng):
    """The Poincaré metric has lambda_p = 2/(1 - c||p||^2). Parallel transport
    should preserve the metric inner product on tangent space."""
    c = 1.0
    for _ in range(10):
        p = random_poincare_point(rng, max_norm=0.6)
        q = random_poincare_point(rng, max_norm=0.6)
        v = rng.normal(scale=0.2, size=p.shape)
        v_t = poincare.parallel_transport(v, p, q, c)
        lambda_p = 2.0 / (1.0 - c * np.dot(p, p))
        lambda_q = 2.0 / (1.0 - c * np.dot(q, q))
        norm_p = lambda_p * np.linalg.norm(v)
        norm_q = lambda_q * np.linalg.norm(v_t)
        assert abs(norm_p - norm_q) < 1e-8


def test_curvature_changes_distance(rng):
    """Larger c should give larger geodesic distances for the same points."""
    p = np.array([0.1, 0.0, 0.0])
    q = np.array([0.4, 0.0, 0.0])
    d_low = poincare.distance(p, q, c=0.5)
    d_high = poincare.distance(p, q, c=2.0)
    assert d_high > d_low
