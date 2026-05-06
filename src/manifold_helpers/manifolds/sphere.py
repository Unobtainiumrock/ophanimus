import numpy as np
from numpy.typing import NDArray

from .._dtype import _f64


def exp_map(p: NDArray[np.floating], v: NDArray[np.floating]) -> NDArray[np.floating]:
    """Exponential map on the unit sphere S^n.

    Maps a tangent vector v at point p to a new point on the sphere
    by walking along the geodesic (great circle) in the direction of v.

    Parameters
    ----------
    p : NDArray
        Point on the sphere (unit vector).
    v : NDArray
        Tangent vector at p (must be orthogonal to p).

    Returns
    -------
    NDArray
        Resulting point on the sphere.
    """
    p, v = _f64(p), _f64(v)
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        return p
    return np.cos(norm_v) * p + np.sin(norm_v) * (v / norm_v)


def log_map(p: NDArray[np.floating], q: NDArray[np.floating]) -> NDArray[np.floating]:
    """Logarithmic map on the unit sphere S^n.

    Inverse of exp_map: given two points on the sphere, returns the
    tangent vector at p that points toward q along the geodesic.

    Parameters
    ----------
    p : NDArray
        Base point on the sphere (unit vector).
    q : NDArray
        Target point on the sphere (unit vector).

    Returns
    -------
    NDArray
        Tangent vector at p pointing toward q.
    """
    p, q = _f64(p), _f64(q)
    # Project q onto the tangent plane at p
    dot = np.dot(p, q)
    dot = np.clip(dot, -1.0, 1.0)
    u = q - dot * p
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-10:
        return np.zeros_like(p)
    theta = np.arccos(dot)
    return theta * (u / norm_u)


def distance(p: NDArray[np.floating], q: NDArray[np.floating]) -> np.floating:
    """Geodesic distance on the unit sphere S^n.

    The arc length of the great circle connecting p and q.

    Parameters
    ----------
    p : NDArray
        Point on the sphere (unit vector).
    q : NDArray
        Point on the sphere (unit vector).

    Returns
    -------
    np.floating
        Geodesic distance (angle in radians).
    """
    p, q = _f64(p), _f64(q)
    dot = np.clip(np.dot(p, q), -1.0, 1.0)
    return np.arccos(dot)


def parallel_transport(v: NDArray[np.floating], p: NDArray[np.floating], q: NDArray[np.floating]) -> NDArray[np.floating]:
    """Parallel transport on the unit sphere S^n.

    Transports tangent vector v at point p to the tangent space at q
    along the geodesic connecting them.

    The geodesic lives in the 2D plane spanned by p and the tangent
    direction toward q. Components of v in that plane get rotated by
    the arc angle; components orthogonal to the plane are unchanged.

    Parameters
    ----------
    v : NDArray
        Tangent vector at p to transport.
    p : NDArray
        Source point on the sphere (unit vector).
    q : NDArray
        Destination point on the sphere (unit vector).

    Returns
    -------
    NDArray
        Transported tangent vector at q.
    """
    v, p, q = _f64(v), _f64(p), _f64(q)
    u = log_map(p, q)
    theta = np.linalg.norm(u)
    if theta < 1e-10:
        return v.copy()
    u_hat = u / theta
    return v + np.dot(u_hat, v) * ((np.cos(theta) - 1) * u_hat - np.sin(theta) * p)
