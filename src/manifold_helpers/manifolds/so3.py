import numpy as np
from numpy.typing import NDArray

from .._dtype import _f64


def _skew(v: NDArray[np.floating]) -> NDArray[np.floating]:
    """Skew-symmetric (hat) matrix from a 3-vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def exp_map(v: NDArray[np.floating]) -> NDArray[np.floating]:
    """Exponential map on SO(3) via Rodrigues' formula.

    Maps a tangent vector v in so(3) (R^3) to a rotation matrix.
    The direction of v is the rotation axis, the norm is the angle.

    Parameters
    ----------
    v : NDArray
        3-vector in the Lie algebra so(3). Shape (3,).

    Returns
    -------
    NDArray
        3x3 rotation matrix in SO(3).
    """
    v = _f64(v)
    theta = np.linalg.norm(v)
    if theta < 1e-10:
        return np.eye(3)
    K = _skew(v / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def log_map(R: NDArray[np.floating]) -> NDArray[np.floating]:
    """Logarithmic map on SO(3).

    Inverse of exp_map: extracts the rotation vector (axis * angle)
    from a rotation matrix.

    Parameters
    ----------
    R : NDArray
        3x3 rotation matrix in SO(3).

    Returns
    -------
    NDArray
        3-vector in so(3). Shape (3,).
    """
    R = _f64(R)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-10:
        return np.zeros(3)

    # Near pi, the standard formula becomes numerically unstable.
    # Use the symmetric part of R to recover the axis instead.
    if np.abs(theta - np.pi) < 1e-6:
        S = (R + R.T) / 2.0 - np.cos(theta) * np.eye(3)
        # The column with largest norm gives the best axis estimate
        col = np.argmax(np.linalg.norm(S, axis=0))
        axis = S[:, col]
        axis = axis / np.linalg.norm(axis)
        return theta * axis

    # Standard case: extract axis from skew-symmetric part
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]])
    return (theta / (2.0 * np.sin(theta))) * axis


def distance(R1: NDArray[np.floating], R2: NDArray[np.floating]) -> np.floating:
    """Geodesic distance on SO(3).

    The bi-invariant Riemannian distance: the angle of the
    relative rotation R1^T @ R2.

    Parameters
    ----------
    R1 : NDArray
        3x3 rotation matrix.
    R2 : NDArray
        3x3 rotation matrix.

    Returns
    -------
    np.floating
        Geodesic distance (rotation angle in radians).
    """
    R1, R2 = _f64(R1), _f64(R2)
    return np.linalg.norm(log_map(R1.T @ R2))


def parallel_transport(w: NDArray[np.floating], R1: NDArray[np.floating], R2: NDArray[np.floating]) -> NDArray[np.floating]:
    """Parallel transport on SO(3) with the bi-invariant metric.

    Transports tangent vector w at R1 to the tangent space at R2
    along the geodesic. For the bi-invariant metric on SO(3), this
    is rotation by half the relative rotation between R1 and R2.

    Parameters
    ----------
    w : NDArray
        Tangent vector at R1 (3-vector in so(3)).
    R1 : NDArray
        Source rotation matrix (3x3).
    R2 : NDArray
        Destination rotation matrix (3x3).

    Returns
    -------
    NDArray
        Transported tangent vector at R2 (3-vector).
    """
    w, R1, R2 = _f64(w), _f64(R1), _f64(R2)
    rel = log_map(R1.T @ R2)
    R_half = exp_map(rel / 2.0)
    return R_half @ w
