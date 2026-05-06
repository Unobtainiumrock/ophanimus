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


def test_parallel_transport_matches_hyperbolic_dispatch(rng):
    """For c=1, poincare.PT now routes through the Lorentz hyperboloid;
    it should agree with ophanimus.hyperbolic.parallel_transport pair-by-pair."""
    from ophanimus import hyperbolic
    for _ in range(20):
        p = random_poincare_point(rng, max_norm=0.6)
        q = random_poincare_point(rng, max_norm=0.6)
        v = rng.normal(scale=0.2, size=p.shape)
        v_direct = poincare.parallel_transport(v, p, q, c=1.0)
        v_dispatch = hyperbolic.parallel_transport(v, p, q)
        assert np.allclose(v_direct, v_dispatch, atol=1e-10)


def test_parallel_transport_round_trip(rng):
    """PT(PT(v, p, q), q, p) ≈ v. The conformal-scaling-only formula failed
    this in any non-trivial 2D+ case; the gyration fix makes it pass."""
    for _ in range(20):
        p = random_poincare_point(rng, max_norm=0.6)
        q = random_poincare_point(rng, max_norm=0.6)
        v = rng.normal(scale=0.2, size=p.shape)
        v_q = poincare.parallel_transport(v, p, q, c=1.0)
        v_back = poincare.parallel_transport(v_q, q, p, c=1.0)
        assert np.allclose(v, v_back, atol=1e-9)


def test_parallel_transport_self_is_identity(rng):
    """PT to the same point is the identity."""
    for _ in range(10):
        p = random_poincare_point(rng)
        v = rng.normal(scale=0.2, size=p.shape)
        v_t = poincare.parallel_transport(v, p, p, c=1.0)
        assert np.allclose(v, v_t, atol=1e-10)
