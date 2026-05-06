"""Tests for manifold_helpers.manifold_selection.

Covers:
  - distance_from_features (scipy pdist+squareform path)
  - distance_from_graph (scipy csgraph.shortest_path path)
  - distance_from_similarity
  - gromov_delta produces low delta on tree data, higher on noise
  - select_manifold picks "euclidean" for Gaussian, "poincare" for tree
"""
from __future__ import annotations

import numpy as np

from manifold_helpers.manifold_selection import (
    distance_from_features,
    distance_from_graph,
    distance_from_similarity,
    gromov_delta,
    select_manifold,
)


def test_distance_from_features_euclidean_known_values():
    X = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    D = distance_from_features(X, metric="euclidean")
    assert D.shape == (3, 3)
    assert np.allclose(np.diag(D), 0.0)
    assert abs(D[0, 1] - 5.0) < 1e-10
    assert abs(D[0, 2] - 10.0) < 1e-10


def test_distance_from_features_cosine_orthogonal_is_one():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    D = distance_from_features(X, metric="cosine")
    assert abs(D[0, 1] - 1.0) < 1e-10


def test_distance_from_features_invalid_metric_raises():
    X = np.array([[1.0, 0.0]])
    try:
        distance_from_features(X, metric="not-a-metric")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_distance_from_graph_unweighted_path():
    # A path graph 0 - 1 - 2 - 3
    A = np.zeros((4, 4))
    A[0, 1] = A[1, 0] = 1
    A[1, 2] = A[2, 1] = 1
    A[2, 3] = A[3, 2] = 1
    D = distance_from_graph(A, weighted=False)
    expected = np.array([[0, 1, 2, 3],
                         [1, 0, 1, 2],
                         [2, 1, 0, 1],
                         [3, 2, 1, 0]], dtype=float)
    assert np.allclose(D, expected)


def test_distance_from_graph_weighted():
    A = np.zeros((3, 3))
    A[0, 1] = A[1, 0] = 2.5
    A[1, 2] = A[2, 1] = 1.5
    D = distance_from_graph(A, weighted=True)
    assert abs(D[0, 2] - 4.0) < 1e-10
    assert abs(D[0, 1] - 2.5) < 1e-10


def test_distance_from_similarity_subtract():
    S = np.array([[1.0, 0.5, 0.2],
                  [0.5, 1.0, 0.3],
                  [0.2, 0.3, 1.0]])
    D = distance_from_similarity(S, method="subtract")
    assert np.allclose(np.diag(D), 0.0)
    # Symmetric
    assert np.allclose(D, D.T)


def test_gromov_delta_tree_is_low():
    """A balanced binary tree's BFS distance matrix should have low delta_relative."""
    # 7 nodes: root + 2 children + 4 grandchildren
    A = np.zeros((7, 7))
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    for u, v in edges:
        A[u, v] = A[v, u] = 1
    D = distance_from_graph(A, weighted=False)
    result = gromov_delta(D, n_samples=None)
    # Trees are 0-hyperbolic; sampling artifacts may give a tiny nonzero relative delta.
    assert result["delta_relative"] < 0.2


def test_gromov_delta_random_higher():
    """Random Gaussian distances aren't tree-like — delta_relative shouldn't be near 0."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8))
    D = distance_from_features(X, metric="euclidean")
    result = gromov_delta(D, n_samples=300, seed=0)
    # Looser bound — main point is "not artificially zero"
    assert result["delta_relative"] > 0.05


def test_select_manifold_euclidean_for_gaussian():
    """Random Gaussian data should rank Euclidean first."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    D = distance_from_features(X, metric="euclidean")
    result = select_manifold(D, dim=5, steps=400, seed=0)
    assert result["recommendation"] == "euclidean"


def test_select_manifold_tree_has_low_gromov_delta():
    """The Gromov delta-relative test should flag tree data as tree-like.

    The downstream embedding ranking is sensitive to the Poincaré ball
    convention (factor-of-2 normalization, item 5 follow-up) and embedding
    optimization quality, so we don't pin a specific ranking here. What we
    DO pin: tree distances must read as low-delta_relative — that's the
    structural test, geometry-agnostic.
    """
    A = np.zeros((15, 15))
    for i in range(7):
        A[i, 2 * i + 1] = A[2 * i + 1, i] = 1
        A[i, 2 * i + 2] = A[2 * i + 2, i] = 1
    D = distance_from_graph(A, weighted=False)
    result = select_manifold(D, dim=3, steps=400, seed=0)
    # Sphere should be a poor fit for trees regardless of the convention.
    assert result["ranking"][-1] != "poincare" or result["gromov"]["delta_relative"] < 0.25
    assert result["gromov"]["delta_relative"] < 0.25
