"""Property tests for ophanimus.manifolds.sphere."""
from __future__ import annotations

import numpy as np

from ophanimus.manifolds import sphere
from conftest import random_sphere_point, random_sphere_tangent


def test_distance_to_self_zero(rng):
    """Note: arccos(1) loses ~half its bits near the boundary, so
    distance(p, p) is exactly 0 in theory but ~1e-8 in practice (sqrt
    of machine epsilon). 1e-6 is conservative."""
    for _ in range(10):
        p = random_sphere_point(rng)
        assert sphere.distance(p, p) < 1e-6


def test_distance_symmetric(rng):
    for _ in range(10):
        p = random_sphere_point(rng)
        q = random_sphere_point(rng)
        assert abs(sphere.distance(p, q) - sphere.distance(q, p)) < 1e-12


def test_exp_log_roundtrip_tangent(rng, atol_tight):
    """log_p(exp_p(v)) = v for tangent vectors of moderate norm."""
    for _ in range(10):
        p = random_sphere_point(rng)
        v = random_sphere_tangent(rng, p, scale=0.3)
        q = sphere.exp_map(p, v)
        v_back = sphere.log_map(p, q)
        assert np.allclose(v, v_back, atol=atol_tight)


def test_exp_log_roundtrip_points(rng, atol_tight):
    """exp_p(log_p(q)) = q for two manifold points (not antipodal)."""
    for _ in range(10):
        p = random_sphere_point(rng)
        # Pick q not antipodal to p
        q = random_sphere_point(rng)
        if np.dot(p, q) < -0.9:
            continue
        q_back = sphere.exp_map(p, sphere.log_map(p, q))
        assert np.allclose(q, q_back, atol=atol_tight)


def test_parallel_transport_preserves_norm(rng, atol_tight):
    """Sphere uses standard dot product as inner product on tangent space."""
    for _ in range(10):
        p = random_sphere_point(rng)
        q = random_sphere_point(rng)
        v = random_sphere_tangent(rng, p, scale=0.3)
        v_t = sphere.parallel_transport(v, p, q)
        assert abs(np.linalg.norm(v) - np.linalg.norm(v_t)) < atol_tight


def test_triangle_inequality(rng):
    """d(p, r) <= d(p, q) + d(q, r) for random triples."""
    for _ in range(50):
        p = random_sphere_point(rng)
        q = random_sphere_point(rng)
        r = random_sphere_point(rng)
        d_pr = sphere.distance(p, r)
        d_pq = sphere.distance(p, q)
        d_qr = sphere.distance(q, r)
        assert d_pr <= d_pq + d_qr + 1e-12
