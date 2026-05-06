import numpy as np
from numpy.typing import NDArray
from typing import Callable

from .._dtype import _f64


def frechet_mean(
    Y: list[NDArray[np.floating]],
    exp_map: Callable,
    log_map: Callable,
    steps: int = 50,
    tol: float = 1e-10,
) -> NDArray[np.floating]:
    """Compute the Fréchet mean of points on a manifold.

    The Fréchet mean is the point that minimizes the sum of squared
    geodesic distances to all data points — the manifold analogue of
    the arithmetic mean.

    Found iteratively: at each step, log-map all points to the current
    estimate's tangent space, compute the Euclidean mean of those tangent
    vectors, and walk in that direction via exp-map. Converges when the
    mean tangent vector is zero (the defining property of the Fréchet mean).

    Parameters
    ----------
    Y : list of NDArray
        Points on the manifold.
    exp_map : callable
        Exponential map: (point, tangent) -> point.
    log_map : callable
        Logarithmic map: (point, point) -> tangent.
    steps : int
        Maximum iterations.
    tol : float
        Stop when the mean tangent norm falls below this.

    Returns
    -------
    NDArray
        The Fréchet mean on the manifold.
    """
    Y = [_f64(y) for y in Y]
    N = len(Y)
    mu = Y[0].copy()

    for _ in range(steps):
        tangents = [log_map(mu, Y[i]) for i in range(N)]
        mean_tangent = np.mean(tangents, axis=0)
        if np.linalg.norm(mean_tangent) < tol:
            break
        mu = exp_map(mu, mean_tangent)

    return mu
