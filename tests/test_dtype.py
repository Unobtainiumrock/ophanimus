"""Verify fp64 promotion at every public entry point.

Pass float32 inputs; assert the output is float64 (or, for distance scalars,
that no fp32 leaks through). The fp64 stability tricks (clip, eps guards,
arctanh clipping) are tuned for float64 — if a function silently runs in
float32, those tricks lose precision in subtle ways.
"""
from __future__ import annotations

import numpy as np

from manifold_helpers.manifolds import poincare, hyperboloid, sphere, so3, spd
from manifold_helpers.algorithms import (
    frechet_mean, geodesic_interpolate, geodesic_path,
    knn_search, rbf_kernel, manifold_velocities,
)
from manifold_helpers.manifold_selection import (
    distance_from_features, distance_from_similarity, distance_from_graph,
)


def _f32(arr):
    return np.asarray(arr, dtype=np.float32)


def test_sphere_exp_map_promotes():
    p = _f32([1.0, 0.0, 0.0])
    v = _f32([0.0, 0.1, 0.0])
    out = sphere.exp_map(p, v)
    assert out.dtype == np.float64


def test_sphere_log_map_promotes():
    p = _f32([1.0, 0.0, 0.0])
    q = _f32([0.0, 1.0, 0.0])
    out = sphere.log_map(p, q)
    assert out.dtype == np.float64


def test_sphere_parallel_transport_promotes():
    p = _f32([1.0, 0.0, 0.0])
    q = _f32([0.0, 1.0, 0.0])
    v = _f32([0.0, 0.1, 0.0])
    out = sphere.parallel_transport(v, p, q)
    assert out.dtype == np.float64


def test_poincare_exp_map_promotes():
    p = _f32([0.1, 0.0])
    v = _f32([0.05, 0.0])
    out = poincare.exp_map(p, v)
    assert out.dtype == np.float64


def test_poincare_distance_promotes():
    """Even scalars should come back as np.float64, not np.float32."""
    p = _f32([0.1, 0.0])
    q = _f32([0.2, 0.1])
    d = poincare.distance(p, q)
    assert isinstance(d, (float, np.floating))
    # The underlying compute should be in fp64
    assert np.asarray(d).dtype == np.float64


def test_hyperboloid_exp_map_promotes():
    p = _f32([np.cosh(0.5), np.sinh(0.5), 0.0])
    v = _f32([np.sinh(0.5) * 0.1, np.cosh(0.5) * 0.1, 0.0])
    out = hyperboloid.exp_map(p, v)
    assert out.dtype == np.float64


def test_hyperboloid_to_poincare_promotes():
    x = _f32([np.cosh(0.5), np.sinh(0.5), 0.0])
    out = hyperboloid.to_poincare(x)
    assert out.dtype == np.float64


def test_hyperboloid_from_poincare_promotes():
    p = _f32([0.3, 0.0])
    out = hyperboloid.from_poincare(p)
    assert out.dtype == np.float64


def test_so3_exp_map_promotes():
    v = _f32([0.1, 0.2, 0.3])
    R = so3.exp_map(v)
    assert R.dtype == np.float64


def test_so3_log_map_promotes():
    R = _f32(np.eye(3))
    out = so3.log_map(R)
    assert out.dtype == np.float64


def test_spd_exp_map_promotes():
    P = _f32(np.eye(3) * 2.0)
    V = _f32(np.eye(3) * 0.1)
    out = spd.exp_map(P, V)
    assert out.dtype == np.float64


def test_spd_distance_promotes():
    P = _f32(np.eye(3) * 2.0)
    Q = _f32(np.eye(3) * 3.0)
    d = spd.distance(P, Q)
    assert np.asarray(d).dtype == np.float64


def test_frechet_mean_promotes():
    pts = [_f32([1.0, 0.0, 0.0]), _f32([0.0, 1.0, 0.0])]
    out = frechet_mean(pts, sphere.exp_map, sphere.log_map)
    assert out.dtype == np.float64


def test_geodesic_interpolate_promotes():
    p = _f32([1.0, 0.0, 0.0])
    q = _f32([0.0, 1.0, 0.0])
    out = geodesic_interpolate(p, q, 0.5, sphere.exp_map, sphere.log_map)
    assert out.dtype == np.float64


def test_geodesic_path_promotes():
    p = _f32([1.0, 0.0, 0.0])
    q = _f32([0.0, 1.0, 0.0])
    pts = geodesic_path(p, q, n_points=5, exp_map=sphere.exp_map, log_map=sphere.log_map)
    for pt in pts:
        assert pt.dtype == np.float64


def test_knn_search_promotes():
    rng = np.random.default_rng(0)
    pts = [_f32(rng.normal(size=3)) for _ in range(5)]
    pts = [p / np.linalg.norm(p) for p in pts]
    q = pts[0]
    idx, dists = knn_search(q, pts, k=2, distance=sphere.distance)
    assert dists.dtype == np.float64


def test_rbf_kernel_promotes():
    rng = np.random.default_rng(0)
    pts = [_f32(rng.normal(size=3)) for _ in range(4)]
    pts = [p / np.linalg.norm(p) for p in pts]
    K = rbf_kernel(pts, sphere.distance, sigma=1.0)
    assert K.dtype == np.float64


def test_manifold_velocities_promotes():
    rng = np.random.default_rng(0)
    traj = [_f32(rng.normal(size=3)) for _ in range(4)]
    traj = [p / np.linalg.norm(p) for p in traj]
    vels = manifold_velocities(traj, sphere.log_map)
    for v in vels:
        assert v.dtype == np.float64


def test_distance_from_features_promotes():
    X = _f32(np.random.RandomState(0).randn(5, 3))
    D = distance_from_features(X, metric="euclidean")
    assert D.dtype == np.float64


def test_distance_from_similarity_promotes():
    S = _f32(np.eye(4) + 0.1 * np.random.RandomState(0).rand(4, 4))
    S = (S + S.T) / 2.0
    D = distance_from_similarity(S, method="subtract")
    assert D.dtype == np.float64


def test_distance_from_graph_promotes():
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32)
    D = distance_from_graph(A, weighted=False)
    assert D.dtype == np.float64
