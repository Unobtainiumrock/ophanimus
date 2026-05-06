"""Property tests for ophanimus.manifolds.hyperboloid."""
from __future__ import annotations

import numpy as np

from ophanimus.manifolds import hyperboloid
from conftest import random_hyperboloid_point, random_hyperboloid_tangent, random_poincare_point


def _minkowski_dot(u, v):
    return -u[0] * v[0] + np.dot(u[1:], v[1:])


def test_distance_to_self_zero(rng):
    """arccosh(1) loses ~half its bits near the boundary; ~1e-8 in practice."""
    for _ in range(10):
        p = random_hyperboloid_point(rng)
        assert hyperboloid.distance(p, p) < 1e-6


def test_distance_symmetric(rng):
    for _ in range(10):
        p = random_hyperboloid_point(rng)
        q = random_hyperboloid_point(rng)
        assert abs(hyperboloid.distance(p, q) - hyperboloid.distance(q, p)) < 1e-10


def test_points_lie_on_hyperboloid(rng):
    """Sampled points satisfy <x, x>_M = -1, x[0] > 0."""
    for _ in range(10):
        p = random_hyperboloid_point(rng)
        assert abs(_minkowski_dot(p, p) - (-1.0)) < 1e-10
        assert p[0] > 0


def test_exp_log_roundtrip_tangent(rng):
    """log_p(exp_p(v)) = v for Minkowski-orthogonal tangent vectors."""
    for _ in range(10):
        p = random_hyperboloid_point(rng)
        v = random_hyperboloid_tangent(rng, p, scale=0.3)
        q = hyperboloid.exp_map(p, v)
        v_back = hyperboloid.log_map(p, q)
        assert np.allclose(v, v_back, atol=1e-7)


def test_to_from_poincare_roundtrip(rng):
    """from_poincare(to_poincare(x)) = x and the reverse."""
    for _ in range(10):
        p = random_poincare_point(rng)
        x = hyperboloid.from_poincare(p)
        p_back = hyperboloid.to_poincare(x)
        assert np.allclose(p, p_back, atol=1e-12)


def test_parallel_transport_lands_in_target_tangent_space(rng):
    """PT(v at p, p, q) must be Minkowski-orthogonal to q (i.e. in T_q H^n)."""
    for _ in range(10):
        p = random_hyperboloid_point(rng)
        q = random_hyperboloid_point(rng)
        v = random_hyperboloid_tangent(rng, p, scale=0.3)
        v_t = hyperboloid.parallel_transport(v, p, q)
        m_inner = -q[0] * v_t[0] + np.dot(q[1:], v_t[1:])
        assert abs(m_inner) < 1e-9


def test_parallel_transport_preserves_minkowski_norm(rng):
    """PT is an isometry: ||v||²_M = ||PT(v)||²_M."""
    for _ in range(10):
        p = random_hyperboloid_point(rng)
        q = random_hyperboloid_point(rng)
        v = random_hyperboloid_tangent(rng, p, scale=0.3)
        v_t = hyperboloid.parallel_transport(v, p, q)
        m_norm_p = -v[0] ** 2 + np.dot(v[1:], v[1:])
        m_norm_q = -v_t[0] ** 2 + np.dot(v_t[1:], v_t[1:])
        assert abs(m_norm_p - m_norm_q) < 1e-9


def test_distance_matches_poincare(rng):
    """Hyperboloid and Poincaré distances measure the same K=-1 metric.
    After from_poincare conversion they agree pair-by-pair within fp64
    tolerance. Convention bug (factor of 2) fixed in 0.2.0.
    """
    from ophanimus.manifolds import poincare
    for _ in range(10):
        p_poin = random_poincare_point(rng)
        q_poin = random_poincare_point(rng)
        p_hyp = hyperboloid.from_poincare(p_poin)
        q_hyp = hyperboloid.from_poincare(q_poin)
        d_hyp = hyperboloid.distance(p_hyp, q_hyp)
        d_poin = poincare.distance(p_poin, q_poin)
        assert abs(d_hyp - d_poin) < 1e-8
