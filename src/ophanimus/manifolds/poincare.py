import numpy as np
from numpy.typing import NDArray

from .._dtype import _f64


def _mobius_add(u: NDArray[np.floating], v: NDArray[np.floating], c: float = 1.0) -> NDArray[np.floating]:
    """Möbius addition in the Poincaré ball with curvature K = -c."""
    u_sq = np.dot(u, u)
    v_sq = np.dot(v, v)
    uv = np.dot(u, v)
    num = (1 + 2 * c * uv + c * v_sq) * u + (1 - c * u_sq) * v
    denom = 1 + 2 * c * uv + c**2 * u_sq * v_sq
    return num / denom


def distance(u: NDArray[np.floating], v: NDArray[np.floating], c: float = 1.0) -> np.floating:
    """Geodesic distance in the Poincaré ball model D^n with curvature K = -c.

    Parameters
    ----------
    u : NDArray
        Point in the Poincaré ball (norm < 1/sqrt(c)).
    v : NDArray
        Point in the Poincaré ball (norm < 1/sqrt(c)).
    c : float
        Curvature parameter (c > 0). Higher c = more curved space.
        c -> 0 recovers Euclidean distance.

    Returns
    -------
    np.floating
        Geodesic distance.
    """
    u, v = _f64(u), _f64(v)
    sqrt_c = np.sqrt(c)
    diff_sq = np.linalg.norm(u - v) ** 2
    u_sq = np.linalg.norm(u) ** 2
    v_sq = np.linalg.norm(v) ** 2
    arg = 1 + 2 * c * diff_sq / ((1 - c * u_sq) * (1 - c * v_sq))
    # The standard K=-c Poincaré ball geodesic distance has prefactor
    # 1/sqrt(c) in arccosh form. The previous implementation used 2/sqrt(c),
    # which is the value of (2/sqrt(c)) * arctanh(sqrt(c)*||M-add||) — the
    # gyrovector form — but the equivalent arccosh form has prefactor
    # 1/sqrt(c) (since arctanh(x) = (1/2) arccosh((1+x²)/(1-x²))).
    return (1.0 / sqrt_c) * np.arccosh(arg)


def exp_map(p: NDArray[np.floating], v: NDArray[np.floating], c: float = 1.0) -> NDArray[np.floating]:
    """Exponential map in the Poincaré ball model D^n with curvature K = -c.

    Parameters
    ----------
    p : NDArray
        Base point in the Poincaré ball (norm < 1/sqrt(c)).
    v : NDArray
        Tangent vector at p (in the Euclidean tangent space).
    c : float
        Curvature parameter (c > 0).

    Returns
    -------
    NDArray
        Resulting point in the Poincaré ball.
    """
    p, v = _f64(p), _f64(v)
    sqrt_c = np.sqrt(c)
    p_sq = np.dot(p, p)
    conformal = 2.0 / (1.0 - c * p_sq)  # lambda_p^c
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        return p
    direction = np.tanh(sqrt_c * conformal * norm_v / 2.0) * (v / (sqrt_c * norm_v))
    return _mobius_add(p, direction, c)


def log_map(p: NDArray[np.floating], q: NDArray[np.floating], c: float = 1.0) -> NDArray[np.floating]:
    """Logarithmic map in the Poincaré ball model D^n with curvature K = -c.

    Parameters
    ----------
    p : NDArray
        Base point in the Poincaré ball (norm < 1/sqrt(c)).
    q : NDArray
        Target point in the Poincaré ball (norm < 1/sqrt(c)).
    c : float
        Curvature parameter (c > 0).

    Returns
    -------
    NDArray
        Tangent vector at p pointing toward q.
    """
    p, q = _f64(p), _f64(q)
    sqrt_c = np.sqrt(c)
    p_sq = np.dot(p, p)
    conformal = 2.0 / (1.0 - c * p_sq)  # lambda_p^c
    diff = _mobius_add(-p, q, c)
    norm_diff = np.linalg.norm(diff)
    if norm_diff < 1e-10:
        return np.zeros_like(p)
    return (2.0 / (sqrt_c * conformal)) * np.arctanh(sqrt_c * norm_diff) * (diff / norm_diff)


def parallel_transport(v: NDArray[np.floating], p: NDArray[np.floating], q: NDArray[np.floating], c: float = 1.0) -> NDArray[np.floating]:
    """Parallel transport in the Poincaré ball model D^n.

    Transports tangent vector v at point p to the tangent space at q
    along the geodesic. Uses the conformal factor ratio, which is
    exact for the Poincaré ball's conformally flat geometry.

    Parameters
    ----------
    v : NDArray
        Tangent vector at p to transport.
    p : NDArray
        Source point in the Poincaré ball.
    q : NDArray
        Destination point in the Poincaré ball.
    c : float
        Curvature parameter (c > 0).

    Returns
    -------
    NDArray
        Transported tangent vector at q.
    """
    v, p, q = _f64(v), _f64(p), _f64(q)
    lambda_p = 2.0 / (1.0 - c * np.dot(p, p))
    lambda_q = 2.0 / (1.0 - c * np.dot(q, q))
    return (lambda_p / lambda_q) * v
