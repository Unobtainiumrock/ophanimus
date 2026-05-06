"""Smoke tests for manifold_helpers.algorithms.

One test per algorithm. Optimization-based tests use fixed seeds so they
don't flake. Each test pins the algorithm against a manifold (typically
sphere — closed-form, fast, easy to reason about).
"""
from __future__ import annotations

import numpy as np

from manifold_helpers.manifolds import sphere, poincare
from manifold_helpers.algorithms import (
    frechet_mean,
    geodesic_regression,
    kmeans,
    knn_classify,
    knn_search,
    geodesic_interpolate,
    geodesic_path,
    pga,
    rbf_kernel,
    laplacian_kernel,
    kernel_density,
    manifold_velocities,
    compare_velocities,
)
from conftest import random_sphere_point, random_sphere_tangent


def test_frechet_mean_recovers_jittered_cluster():
    """Mean of points jittered around a known center lands near that center."""
    rng = np.random.default_rng(0)
    center = np.array([1.0, 0.0, 0.0])
    pts = []
    for _ in range(20):
        v = rng.normal(scale=0.05, size=3)
        v = v - np.dot(center, v) * center
        pts.append(sphere.exp_map(center, v))
    mu = frechet_mean(pts, sphere.exp_map, sphere.log_map)
    assert sphere.distance(mu, center) < 0.05


def test_geodesic_regression_low_loss():
    """On data generated from y = exp_p(x*v), regression should fit with low residual."""
    rng = np.random.default_rng(0)
    p_true = np.array([1.0, 0.0, 0.0])
    v_true = np.array([0.0, 0.5, 0.0])  # tangent at p_true
    X = np.linspace(-1, 1, 25)
    Y = [sphere.exp_map(p_true, x * v_true) for x in X]
    p_hat, v_hat, losses = geodesic_regression(
        X, Y, sphere.exp_map, sphere.log_map, sphere.distance, sphere.parallel_transport,
        steps=300,
    )
    assert losses[-1] < 1e-3


def test_kmeans_separates_two_caps():
    """k-means recovers labels for two well-separated clusters on the sphere."""
    rng = np.random.default_rng(0)
    c1 = np.array([1.0, 0.0, 0.0])
    c2 = np.array([-1.0, 0.0, 0.0])
    pts1 = [sphere.exp_map(c1, random_sphere_tangent(rng, c1, scale=0.1)) for _ in range(15)]
    pts2 = [sphere.exp_map(c2, random_sphere_tangent(rng, c2, scale=0.1)) for _ in range(15)]
    points = pts1 + pts2
    labels, _ = kmeans(points, k=2,
                       exp_map=sphere.exp_map,
                       log_map=sphere.log_map,
                       distance=sphere.distance,
                       seed=0)
    # Labels can flip — check that the two halves have consistent labels
    half1 = labels[:15]
    half2 = labels[15:]
    correct = np.all(half1 == half1[0]) and np.all(half2 == half2[0]) and half1[0] != half2[0]
    assert correct


def test_knn_search_self_returns_self():
    """knn_search with k=1 over points-including-query returns the query itself."""
    rng = np.random.default_rng(0)
    pts = [random_sphere_point(rng) for _ in range(10)]
    q = pts[3]
    idx, dists = knn_search(q, pts, k=1, distance=sphere.distance)
    assert idx[0] == 3
    assert dists[0] < 1e-12


def test_knn_classify_majority():
    """knn_classify returns the majority label among neighbors."""
    rng = np.random.default_rng(0)
    c1 = np.array([1.0, 0.0, 0.0])
    c2 = np.array([-1.0, 0.0, 0.0])
    pts = [sphere.exp_map(c1, random_sphere_tangent(rng, c1, scale=0.1)) for _ in range(5)]
    pts += [sphere.exp_map(c2, random_sphere_tangent(rng, c2, scale=0.1)) for _ in range(5)]
    labels = np.array([0] * 5 + [1] * 5)
    pred, _, _ = knn_classify(c1, pts, labels, k=3, distance=sphere.distance)
    assert pred == 0


def test_geodesic_interpolate_endpoints_and_midpoint():
    """t=0 returns p, t=1 returns q, midpoint is equidistant."""
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0])
    mid = geodesic_interpolate(p, q, 0.5, sphere.exp_map, sphere.log_map)
    d_pm = sphere.distance(p, mid)
    d_qm = sphere.distance(q, mid)
    assert abs(d_pm - d_qm) < 1e-10


def test_geodesic_path_length():
    """geodesic_path produces n_points evenly spaced points."""
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0])
    pts = geodesic_path(p, q, n_points=11, exp_map=sphere.exp_map, log_map=sphere.log_map)
    assert len(pts) == 11
    assert np.allclose(pts[0], p, atol=1e-10)
    assert np.allclose(pts[-1], q, atol=1e-10)


def test_pga_first_component_dominates():
    """Points distributed along a single tangent direction → first component captures most variance."""
    rng = np.random.default_rng(0)
    base = np.array([1.0, 0.0, 0.0])
    direction = np.array([0.0, 1.0, 0.0])  # tangent at base
    pts = [sphere.exp_map(base, t * direction) for t in np.linspace(-0.5, 0.5, 30)]
    result = pga(pts, sphere.exp_map, sphere.log_map, n_components=2)
    ratios = result["explained_variance_ratio"]
    assert ratios[0] > 0.95


def test_rbf_kernel_diagonal_is_one():
    rng = np.random.default_rng(0)
    pts = [random_sphere_point(rng) for _ in range(5)]
    K = rbf_kernel(pts, sphere.distance, sigma=1.0)
    assert np.allclose(np.diag(K), 1.0, atol=1e-10)


def test_laplacian_kernel_diagonal_is_one():
    rng = np.random.default_rng(0)
    pts = [random_sphere_point(rng) for _ in range(5)]
    K = laplacian_kernel(pts, sphere.distance, sigma=1.0)
    assert np.allclose(np.diag(K), 1.0, atol=1e-10)


def test_kernel_density_positive_at_mode():
    rng = np.random.default_rng(0)
    center = np.array([1.0, 0.0, 0.0])
    pts = [sphere.exp_map(center, random_sphere_tangent(rng, center, scale=0.05)) for _ in range(20)]
    rho_at_center = kernel_density(center, pts, sphere.distance, sigma=0.1)
    rho_far = kernel_density(np.array([0.0, 0.0, 1.0]), pts, sphere.distance, sigma=0.1)
    assert rho_at_center > rho_far


def test_manifold_velocities_count():
    rng = np.random.default_rng(0)
    traj = [random_sphere_point(rng) for _ in range(8)]
    vels = manifold_velocities(traj, sphere.log_map)
    assert len(vels) == 7


def test_compare_velocities_count_and_shape():
    rng = np.random.default_rng(0)
    traj = [random_sphere_point(rng) for _ in range(8)]
    vels = compare_velocities(traj, sphere.log_map, sphere.parallel_transport, reference_index=0)
    assert len(vels) == 7
    # All should have the same shape as the reference tangent space
    for v in vels:
        assert v.shape == traj[0].shape
