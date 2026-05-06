import numpy as np
from numpy.typing import NDArray
from typing import Callable
from .frechet_mean import frechet_mean
from .._dtype import _f64


def kmeans(
    points: list[NDArray[np.floating]],
    k: int,
    exp_map: Callable,
    log_map: Callable,
    distance: Callable,
    steps: int = 100,
    n_init: int = 5,
    seed: int = 42,
) -> tuple[NDArray, list[NDArray[np.floating]]]:
    """k-means clustering on a Riemannian manifold.

    Manifold analogue of Lloyd's algorithm:
    1. Assign each point to the nearest center (using geodesic distance).
    2. Recompute each center as the Fréchet mean of its cluster.
    3. Repeat until convergence.

    Parameters
    ----------
    points : list of NDArray
        Data points on the manifold.
    k : int
        Number of clusters.
    exp_map : callable
        Exponential map: (point, tangent) -> point.
    log_map : callable
        Logarithmic map: (point, point) -> tangent.
    distance : callable
        Geodesic distance: (point, point) -> scalar.
    steps : int
        Maximum iterations per run.
    n_init : int
        Number of random initializations (best result is returned).
    seed : int
        Random seed.

    Returns
    -------
    labels : NDArray
        Cluster assignment for each point, shape (N,).
    centers : list of NDArray
        Fréchet mean of each cluster.
    """
    points = [_f64(p) for p in points]
    N = len(points)
    rng = np.random.default_rng(seed)

    best_labels = None
    best_centers = None
    best_inertia = np.inf

    for init in range(n_init):
        # Random initialization: pick k distinct points as centers
        idx = rng.choice(N, size=k, replace=False)
        centers = [points[i].copy() for i in idx]

        labels = np.zeros(N, dtype=int)

        for _ in range(steps):
            # Assign each point to nearest center
            new_labels = np.zeros(N, dtype=int)
            for i in range(N):
                dists = [distance(points[i], centers[j]) for j in range(k)]
                new_labels[i] = int(np.argmin(dists))

            # Check convergence
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

            # Recompute centers as Fréchet means
            for j in range(k):
                members = [points[i] for i in range(N) if labels[i] == j]
                if len(members) > 0:
                    centers[j] = frechet_mean(members, exp_map, log_map)

        # Compute inertia (total within-cluster distance)
        inertia = sum(distance(points[i], centers[labels[i]]) ** 2 for i in range(N))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centers = centers

    return best_labels, best_centers
