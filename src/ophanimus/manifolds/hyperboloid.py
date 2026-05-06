import numpy as np
from numpy.typing import NDArray

from .._dtype import _f64


def _minkowski_dot(u: NDArray[np.floating], v: NDArray[np.floating]) -> np.floating:
    """Minkowski inner product: -u0*v0 + u1*v1 + ... + un*vn."""
    return -u[0] * v[0] + np.dot(u[1:], v[1:])


def distance(u: NDArray[np.floating], v: NDArray[np.floating]) -> np.floating:
    """Geodesic distance on the hyperboloid H^n.

    Parameters
    ----------
    u : NDArray
        Point on the hyperboloid, shape (n+1,).
        Satisfies -u0^2 + u1^2 + ... + un^2 = -1, u0 > 0.
    v : NDArray
        Point on the hyperboloid, shape (n+1,).

    Returns
    -------
    np.floating
        Geodesic distance.
    """
    u, v = _f64(u), _f64(v)
    inner = -_minkowski_dot(u, v)
    return np.arccosh(np.maximum(inner, 1.0))


def exp_map(p: NDArray[np.floating], v: NDArray[np.floating]) -> NDArray[np.floating]:
    """Exponential map on the hyperboloid H^n.

    Parameters
    ----------
    p : NDArray
        Base point on the hyperboloid, shape (n+1,).
    v : NDArray
        Tangent vector at p. Must satisfy <p, v>_M = 0
        (Minkowski-orthogonal to p).

    Returns
    -------
    NDArray
        Resulting point on the hyperboloid.
    """
    p, v = _f64(p), _f64(v)
    norm_v = np.sqrt(np.maximum(_minkowski_dot(v, v), 0.0))
    if norm_v < 1e-10:
        return p.copy()
    return np.cosh(norm_v) * p + np.sinh(norm_v) * (v / norm_v)


def log_map(p: NDArray[np.floating], q: NDArray[np.floating]) -> NDArray[np.floating]:
    """Logarithmic map on the hyperboloid H^n.

    Parameters
    ----------
    p : NDArray
        Base point on the hyperboloid, shape (n+1,).
    q : NDArray
        Target point on the hyperboloid, shape (n+1,).

    Returns
    -------
    NDArray
        Tangent vector at p pointing toward q.
    """
    p, q = _f64(p), _f64(q)
    inner = -_minkowski_dot(p, q)
    inner = np.maximum(inner, 1.0)
    d = np.arccosh(inner)
    if d < 1e-10:
        return np.zeros_like(p)
    # Project q onto tangent space at p, then scale
    u = q + _minkowski_dot(p, q) * p  # = q - <p,q>_M * p (note sign)
    norm_u = np.sqrt(np.maximum(_minkowski_dot(u, u), 1e-15))
    return d * (u / norm_u)


def parallel_transport(v: NDArray[np.floating], p: NDArray[np.floating],
                       q: NDArray[np.floating]) -> NDArray[np.floating]:
    """Parallel transport on the hyperboloid H^n.

    Transports tangent vector v at p to the tangent space at q
    along the geodesic.

    Parameters
    ----------
    v : NDArray
        Tangent vector at p to transport.
    p : NDArray
        Source point on the hyperboloid.
    q : NDArray
        Destination point on the hyperboloid.

    Returns
    -------
    NDArray
        Transported tangent vector at q.
    """
    v, p, q = _f64(v), _f64(p), _f64(q)
    d = distance(p, q)
    if d < 1e-10:
        return v.copy()
    log_pq = log_map(p, q)
    log_pq_hat = log_pq / d
    # v_t = v + <log_pq_hat, v>_M * (-sinh(d)*p + (cosh(d)-1)*log_pq_hat)...
    # but in Minkowski metric. The clean formula:
    inner = _minkowski_dot(log_pq_hat, v)
    return v + inner * ((np.cosh(d) - 1) * log_pq_hat - np.sinh(d) * p)


# ---------------------------------------------------------------------------
# Coordinate conversions: Hyperboloid <-> Poincaré ball
# ---------------------------------------------------------------------------

def to_poincare(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert hyperboloid coordinates to Poincaré ball coordinates.

    Maps a point on H^n (shape n+1) to the Poincaré ball D^n (shape n).
    Uses stereographic projection from the south pole.

    Parameters
    ----------
    x : NDArray
        Point on the hyperboloid, shape (n+1,). x[0] is the time component.

    Returns
    -------
    NDArray
        Point in the Poincaré ball, shape (n,).
    """
    x = _f64(x)
    return x[1:] / (1.0 + x[0])


def from_poincare(p: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert Poincaré ball coordinates to hyperboloid coordinates.

    Maps a point in D^n (shape n) to the hyperboloid H^n (shape n+1).

    Parameters
    ----------
    p : NDArray
        Point in the Poincaré ball, shape (n,). norm < 1.

    Returns
    -------
    NDArray
        Point on the hyperboloid, shape (n+1,).
    """
    p = _f64(p)
    p_sq = np.dot(p, p)
    x0 = (1.0 + p_sq) / (1.0 - p_sq)
    xi = 2.0 * p / (1.0 - p_sq)
    return np.concatenate([[x0], xi])
