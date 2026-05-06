import numpy as np
from numpy.typing import NDArray
from typing import Callable
from collections import Counter

from .._dtype import _f64


def knn_classify(
    query: NDArray[np.floating],
    points: list[NDArray[np.floating]],
    labels: NDArray,
    k: int,
    distance: Callable,
) -> tuple:
    """k-nearest neighbor classification on a Riemannian manifold.

    Finds the k closest points (by geodesic distance) and returns
    the majority label.

    Parameters
    ----------
    query : NDArray
        Point on the manifold to classify.
    points : list of NDArray
        Training points on the manifold.
    labels : NDArray
        Labels for each training point.
    k : int
        Number of neighbors.
    distance : callable
        Geodesic distance: (point, point) -> scalar.

    Returns
    -------
    prediction : any
        The majority label among k nearest neighbors.
    neighbor_indices : NDArray
        Indices of the k nearest neighbors.
    neighbor_distances : NDArray
        Geodesic distances to the k nearest neighbors.
    """
    query = _f64(query)
    points = [_f64(p) for p in points]
    dists = np.array([distance(query, p) for p in points])
    idx = np.argsort(dists)[:k]

    neighbor_labels = labels[idx]
    counts = Counter(neighbor_labels)
    prediction = counts.most_common(1)[0][0]

    return prediction, idx, dists[idx]


def knn_search(
    query: NDArray[np.floating],
    points: list[NDArray[np.floating]],
    k: int,
    distance: Callable,
) -> tuple[NDArray, NDArray[np.floating]]:
    """k-nearest neighbor search on a Riemannian manifold.

    Finds the k closest points by geodesic distance (no labels needed).

    Parameters
    ----------
    query : NDArray
        Point on the manifold.
    points : list of NDArray
        Points to search over.
    k : int
        Number of neighbors.
    distance : callable
        Geodesic distance: (point, point) -> scalar.

    Returns
    -------
    indices : NDArray
        Indices of the k nearest neighbors.
    distances : NDArray
        Geodesic distances to the k nearest neighbors.
    """
    query = _f64(query)
    points = [_f64(p) for p in points]
    dists = np.array([distance(query, p) for p in points])
    idx = np.argsort(dists)[:k]
    return idx, dists[idx]
