import numpy as np
from numpy.typing import NDArray
from typing import Callable

from .._dtype import _f64


def rbf_kernel(
    points: list[NDArray[np.floating]],
    distance: Callable,
    sigma: float = 1.0,
) -> NDArray[np.floating]:
    """Geodesic RBF (Gaussian) kernel matrix.

    K(p, q) = exp(-d(p, q)^2 / (2 * sigma^2))

    Drop-in replacement for Euclidean RBF kernel, but uses geodesic
    distance. Works with SVM, Gaussian processes, kernel density
    estimation, etc.

    Parameters
    ----------
    points : list of NDArray
        Points on the manifold.
    distance : callable
        Geodesic distance: (point, point) -> scalar.
    sigma : float
        Bandwidth parameter. Larger sigma = smoother kernel.

    Returns
    -------
    NDArray
        Kernel matrix, shape (N, N). Symmetric, positive values.
    """
    points = [_f64(p) for p in points]
    N = len(points)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            d = distance(points[i], points[j])
            K[i, j] = np.exp(-d ** 2 / (2 * sigma ** 2))
            K[j, i] = K[i, j]
    return K


def laplacian_kernel(
    points: list[NDArray[np.floating]],
    distance: Callable,
    sigma: float = 1.0,
) -> NDArray[np.floating]:
    """Geodesic Laplacian kernel matrix.

    K(p, q) = exp(-d(p, q) / sigma)

    Sharper than RBF — decays linearly in distance rather than
    quadratically. Better for data with sharp cluster boundaries.

    Parameters
    ----------
    points : list of NDArray
        Points on the manifold.
    distance : callable
        Geodesic distance: (point, point) -> scalar.
    sigma : float
        Bandwidth parameter.

    Returns
    -------
    NDArray
        Kernel matrix, shape (N, N).
    """
    points = [_f64(p) for p in points]
    N = len(points)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            d = distance(points[i], points[j])
            K[i, j] = np.exp(-d / sigma)
            K[j, i] = K[i, j]
    return K


def kernel_density(
    query: NDArray[np.floating],
    points: list[NDArray[np.floating]],
    distance: Callable,
    sigma: float = 1.0,
) -> float:
    """Geodesic kernel density estimate at a query point.

    KDE(q) = (1/N) * sum_i K(q, p_i)

    Uses the RBF kernel. Estimates how "dense" the data is around
    the query point on the manifold.

    Parameters
    ----------
    query : NDArray
        Point on the manifold to evaluate density at.
    points : list of NDArray
        Data points on the manifold.
    distance : callable
        Geodesic distance: (point, point) -> scalar.
    sigma : float
        Bandwidth parameter.

    Returns
    -------
    float
        Density estimate at the query point.
    """
    query = _f64(query)
    points = [_f64(p) for p in points]
    N = len(points)
    total = 0.0
    for p in points:
        d = distance(query, p)
        total += np.exp(-d ** 2 / (2 * sigma ** 2))
    return total / N
