import numpy as np
from numpy.typing import NDArray
from typing import Callable

from .._dtype import _f64


def geodesic_interpolate(
    p: NDArray[np.floating],
    q: NDArray[np.floating],
    t: float,
    exp_map: Callable,
    log_map: Callable,
) -> NDArray[np.floating]:
    """Interpolate along the geodesic between two manifold points.

    Returns the point at fraction t along the geodesic from p to q.
    t=0 gives p, t=1 gives q, t=0.5 gives the midpoint.

    Parameters
    ----------
    p : NDArray
        Start point on the manifold.
    q : NDArray
        End point on the manifold.
    t : float
        Interpolation parameter in [0, 1].
    exp_map : callable
        Exponential map: (point, tangent) -> point.
    log_map : callable
        Logarithmic map: (point, point) -> tangent.

    Returns
    -------
    NDArray
        Interpolated point on the manifold.
    """
    p, q = _f64(p), _f64(q)
    v = log_map(p, q)
    return exp_map(p, t * v)


def geodesic_path(
    p: NDArray[np.floating],
    q: NDArray[np.floating],
    n_points: int,
    exp_map: Callable,
    log_map: Callable,
) -> list[NDArray[np.floating]]:
    """Generate evenly spaced points along the geodesic from p to q.

    Parameters
    ----------
    p : NDArray
        Start point on the manifold.
    q : NDArray
        End point on the manifold.
    n_points : int
        Number of points to generate (including endpoints).
    exp_map : callable
        Exponential map: (point, tangent) -> point.
    log_map : callable
        Logarithmic map: (point, point) -> tangent.

    Returns
    -------
    list of NDArray
        Points along the geodesic, evenly spaced in parameter t.
    """
    p, q = _f64(p), _f64(q)
    v = log_map(p, q)
    ts = np.linspace(0, 1, n_points)
    return [exp_map(p, t * v) for t in ts]
