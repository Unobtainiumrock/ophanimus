"""Tests for ophanimus.hyperbolic — the engine/dashboard split.

Three classes of test:
  1. Agreement with poincare.* primitives (where they're known correct)
  2. Mathematical invariants (round-trips, metric preservation)
  3. Boundary stability (the whole point of the split)
"""
from __future__ import annotations

import numpy as np
import pytest

from ophanimus import hyperbolic
from ophanimus.manifolds import poincare, hyperboloid
from ophanimus.algorithms import frechet_mean, kmeans
from conftest import random_poincare_point


# ---------------------------------------------------------------------------
# Tangent-space conversions
# ---------------------------------------------------------------------------

def test_push_pull_tangent_roundtrip(rng):
    """_pull(_push(v)) = v exactly (modulo fp64 noise)."""
    for _ in range(20):
        p = random_poincare_point(rng)
        v = rng.normal(scale=0.2, size=p.shape)
        v_back = hyperbolic._pull_tangent(hyperbolic._push_tangent(v, p), p)
        assert np.allclose(v, v_back, atol=1e-12)


def test_push_tangent_is_minkowski_orthogonal(rng):
    """_push_tangent(v, p) lies in T_{phi(p)} H^n (Minkowski-orthogonal to phi(p))."""
    for _ in range(20):
        p_poin = random_poincare_point(rng)
        v_poin = rng.normal(scale=0.2, size=p_poin.shape)
        p_hyp = hyperboloid.from_poincare(p_poin)
        v_hyp = hyperbolic._push_tangent(v_poin, p_poin)
        # Minkowski inner product of point and tangent must be 0
        m_inner = -p_hyp[0] * v_hyp[0] + np.dot(p_hyp[1:], v_hyp[1:])
        assert abs(m_inner) < 1e-10


def test_push_tangent_preserves_metric_norm(rng):
    """The Riemannian norm of v_poin equals the Minkowski norm of _push(v_poin)."""
    for _ in range(20):
        p = random_poincare_point(rng)
        v = rng.normal(scale=0.2, size=p.shape)
        # Riemannian (Poincaré) squared norm: lambda_p² |v|² where lambda_p = 2/(1-|p|²)
        lambda_p = 2.0 / (1.0 - np.dot(p, p))
        riem_norm_sq = (lambda_p ** 2) * np.dot(v, v)
        v_hyp = hyperbolic._push_tangent(v, p)
        # Minkowski squared norm
        m_norm_sq = -v_hyp[0] ** 2 + np.dot(v_hyp[1:], v_hyp[1:])
        assert abs(riem_norm_sq - m_norm_sq) < 1e-10


# ---------------------------------------------------------------------------
# Distance + exp/log agree with the (now-correct) Poincaré primitives
# ---------------------------------------------------------------------------

def test_distance_matches_poincare(rng):
    """hyperbolic.distance == poincare.distance after the 0.2.0 convention fix."""
    for _ in range(20):
        p = random_poincare_point(rng)
        q = random_poincare_point(rng)
        d_h = hyperbolic.distance(p, q)
        d_p = poincare.distance(p, q)
        assert abs(d_h - d_p) < 1e-9


def test_exp_map_matches_poincare(rng):
    """hyperbolic.exp_map ≈ poincare.exp_map."""
    for _ in range(20):
        p = random_poincare_point(rng, max_norm=0.5)
        v = rng.normal(scale=0.2, size=p.shape)
        q_h = hyperbolic.exp_map(p, v)
        q_p = poincare.exp_map(p, v)
        assert np.allclose(q_h, q_p, atol=1e-9)


def test_log_map_matches_poincare(rng):
    """hyperbolic.log_map ≈ poincare.log_map."""
    for _ in range(20):
        p = random_poincare_point(rng, max_norm=0.5)
        q = random_poincare_point(rng, max_norm=0.5)
        v_h = hyperbolic.log_map(p, q)
        v_p = poincare.log_map(p, q)
        assert np.allclose(v_h, v_p, atol=1e-9)


# ---------------------------------------------------------------------------
# Round-trip invariants
# ---------------------------------------------------------------------------

def test_exp_log_roundtrip(rng):
    """log_p(exp_p(v)) = v in Poincaré coords."""
    for _ in range(20):
        p = random_poincare_point(rng, max_norm=0.5)
        v = rng.normal(scale=0.2, size=p.shape)
        q = hyperbolic.exp_map(p, v)
        v_back = hyperbolic.log_map(p, q)
        assert np.allclose(v, v_back, atol=1e-9)


def test_distance_self_zero(rng):
    """distance(p, p) ≈ 0."""
    for _ in range(10):
        p = random_poincare_point(rng)
        assert hyperbolic.distance(p, p) < 1e-6


def test_distance_symmetric(rng):
    """distance(p, q) == distance(q, p)."""
    for _ in range(10):
        p = random_poincare_point(rng)
        q = random_poincare_point(rng)
        assert abs(hyperbolic.distance(p, q) - hyperbolic.distance(q, p)) < 1e-12


def test_triangle_inequality(rng):
    """d(p, r) <= d(p, q) + d(q, r) for random triples."""
    for _ in range(50):
        p = random_poincare_point(rng)
        q = random_poincare_point(rng)
        r = random_poincare_point(rng)
        d_pr = hyperbolic.distance(p, r)
        d_pq = hyperbolic.distance(p, q)
        d_qr = hyperbolic.distance(q, r)
        assert d_pr <= d_pq + d_qr + 1e-10


# ---------------------------------------------------------------------------
# Parallel transport — preserves the metric inner product. Does NOT need to
# agree with poincare.parallel_transport (which is the conformal-only
# approximation, missing the Möbius gyration — see HE-PT-MOBIUS-GYRATION).
# ---------------------------------------------------------------------------

def test_parallel_transport_preserves_metric_norm(rng):
    """||PT(v, p, q)||_{g_q} = ||v||_{g_p} — true PT along a geodesic."""
    for _ in range(20):
        p = random_poincare_point(rng, max_norm=0.5)
        q = random_poincare_point(rng, max_norm=0.5)
        v = rng.normal(scale=0.2, size=p.shape)
        v_t = hyperbolic.parallel_transport(v, p, q)
        # Riemannian norm: lambda_x² |v|²
        lambda_p = 2.0 / (1.0 - np.dot(p, p))
        lambda_q = 2.0 / (1.0 - np.dot(q, q))
        norm_p = lambda_p * np.linalg.norm(v)
        norm_q = lambda_q * np.linalg.norm(v_t)
        assert abs(norm_p - norm_q) < 1e-9


def test_parallel_transport_identity_when_p_equals_q(rng):
    """PT to the same point is the identity."""
    for _ in range(10):
        p = random_poincare_point(rng)
        v = rng.normal(scale=0.2, size=p.shape)
        v_t = hyperbolic.parallel_transport(v, p, p)
        assert np.allclose(v, v_t, atol=1e-10)


# ---------------------------------------------------------------------------
# Boundary stability — the engine/dashboard split's payoff
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("radius", [0.9, 0.99, 0.999, 0.9999])
def test_distance_finite_near_boundary(radius):
    """hyperbolic.distance produces finite, non-NaN values near the Poincaré boundary."""
    p = np.array([radius, 0.0])
    q = np.array([-radius, 0.0])
    d = hyperbolic.distance(p, q)
    assert np.isfinite(d)
    assert d > 0


@pytest.mark.parametrize("radius", [0.9, 0.99, 0.999])
def test_exp_log_finite_near_boundary(radius):
    """exp/log at points near the Poincaré boundary don't NaN."""
    p = np.array([radius, 0.0])
    q = np.array([0.0, radius])
    v = hyperbolic.log_map(p, q)
    q_back = hyperbolic.exp_map(p, v)
    assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(q_back))


# ---------------------------------------------------------------------------
# End-to-end: algorithms work with hyperbolic.* dispatch
# ---------------------------------------------------------------------------

def test_frechet_mean_via_hyperbolic_dispatch(rng):
    """frechet_mean using hyperbolic.* recovers a known cluster center."""
    center = np.array([0.2, 0.1])
    pts = []
    for _ in range(20):
        v = rng.normal(scale=0.05, size=2)
        pts.append(hyperbolic.exp_map(center, v))
    mu = frechet_mean(pts, hyperbolic.exp_map, hyperbolic.log_map)
    assert hyperbolic.distance(mu, center) < 0.05


def test_kmeans_via_hyperbolic_dispatch(rng):
    """kmeans using hyperbolic.* separates two well-separated clusters."""
    c1 = np.array([0.4, 0.0])
    c2 = np.array([-0.4, 0.0])
    pts1 = [hyperbolic.exp_map(c1, rng.normal(scale=0.05, size=2)) for _ in range(15)]
    pts2 = [hyperbolic.exp_map(c2, rng.normal(scale=0.05, size=2)) for _ in range(15)]
    points = pts1 + pts2
    labels, _ = kmeans(
        points, k=2,
        exp_map=hyperbolic.exp_map,
        log_map=hyperbolic.log_map,
        distance=hyperbolic.distance,
        seed=0,
    )
    half1, half2 = labels[:15], labels[15:]
    assert np.all(half1 == half1[0])
    assert np.all(half2 == half2[0])
    assert half1[0] != half2[0]


# ---------------------------------------------------------------------------
# fp64 boundary preservation — hyperbolic.* should still promote inputs
# ---------------------------------------------------------------------------

def test_dispatch_promotes_fp32_to_fp64():
    p = np.array([0.1, 0.2], dtype=np.float32)
    v = np.array([0.05, -0.03], dtype=np.float32)
    q = hyperbolic.exp_map(p, v)
    assert q.dtype == np.float64
    assert hyperbolic.log_map(p, q).dtype == np.float64
    assert np.asarray(hyperbolic.distance(p, q)).dtype == np.float64
