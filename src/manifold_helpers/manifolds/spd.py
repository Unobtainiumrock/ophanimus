import numpy as np
from numpy.typing import NDArray

from .._dtype import _f64


def _symm(M: NDArray[np.floating]) -> NDArray[np.floating]:
    """Numerical-drift symmetrization guard for SPD inputs.

    SPD matrices are symmetric by definition, but inputs that have been
    propagated through long compute chains drift slightly off-symmetric.
    `_sym_*` helpers below use `eigh` which assumes symmetry; passing a
    drifted matrix gives subtly wrong eigenvalues. One line, no perf cost.
    """
    return (M + M.T) / 2.0


def _sym_sqrt(S: NDArray[np.floating]) -> NDArray[np.floating]:
    """Matrix square root of a symmetric positive definite matrix."""
    eigvals, eigvecs = np.linalg.eigh(S)
    return eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0))) @ eigvecs.T


def _sym_inv_sqrt(S: NDArray[np.floating]) -> NDArray[np.floating]:
    """Inverse matrix square root of a symmetric positive definite matrix."""
    eigvals, eigvecs = np.linalg.eigh(S)
    return eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-15))) @ eigvecs.T


def _sym_logm(S: NDArray[np.floating]) -> NDArray[np.floating]:
    """Matrix logarithm of a symmetric positive definite matrix."""
    eigvals, eigvecs = np.linalg.eigh(S)
    return eigvecs @ np.diag(np.log(np.maximum(eigvals, 1e-15))) @ eigvecs.T


def _sym_expm(S: NDArray[np.floating]) -> NDArray[np.floating]:
    """Matrix exponential of a symmetric matrix."""
    eigvals, eigvecs = np.linalg.eigh(S)
    return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T


def exp_map(P: NDArray[np.floating], V: NDArray[np.floating]) -> NDArray[np.floating]:
    """Exponential map on the SPD manifold Sym+_n.

    Maps a tangent vector V (a symmetric matrix) at base point P
    to a new SPD matrix by walking along the geodesic.

    Parameters
    ----------
    P : NDArray
        Base point, n×n symmetric positive definite matrix.
    V : NDArray
        Tangent vector at P (symmetric matrix).

    Returns
    -------
    NDArray
        Resulting n×n SPD matrix.
    """
    P, V = _f64(P), _f64(V)
    P = _symm(P)
    V = _symm(V)
    P_sqrt = _sym_sqrt(P)
    P_inv_sqrt = _sym_inv_sqrt(P)
    return P_sqrt @ _sym_expm(P_inv_sqrt @ V @ P_inv_sqrt) @ P_sqrt


def log_map(P: NDArray[np.floating], Q: NDArray[np.floating]) -> NDArray[np.floating]:
    """Logarithmic map on the SPD manifold Sym+_n.

    Inverse of exp_map: returns the tangent vector at P pointing
    toward Q along the geodesic.

    Parameters
    ----------
    P : NDArray
        Base point, n×n SPD matrix.
    Q : NDArray
        Target point, n×n SPD matrix.

    Returns
    -------
    NDArray
        Symmetric matrix (tangent vector at P).
    """
    P, Q = _f64(P), _f64(Q)
    P = _symm(P)
    Q = _symm(Q)
    P_sqrt = _sym_sqrt(P)
    P_inv_sqrt = _sym_inv_sqrt(P)
    return P_sqrt @ _sym_logm(P_inv_sqrt @ Q @ P_inv_sqrt) @ P_sqrt


def distance(P: NDArray[np.floating], Q: NDArray[np.floating]) -> np.floating:
    """Affine-invariant Riemannian distance on the SPD manifold.

    Parameters
    ----------
    P : NDArray
        n×n SPD matrix.
    Q : NDArray
        n×n SPD matrix.

    Returns
    -------
    np.floating
        Geodesic distance.
    """
    P, Q = _f64(P), _f64(Q)
    P = _symm(P)
    Q = _symm(Q)
    P_inv_sqrt = _sym_inv_sqrt(P)
    M = P_inv_sqrt @ Q @ P_inv_sqrt
    eigvals = np.linalg.eigvalsh(_symm(M))
    return np.sqrt(np.sum(np.log(np.maximum(eigvals, 1e-15)) ** 2))


def parallel_transport(V: NDArray[np.floating], P: NDArray[np.floating], Q: NDArray[np.floating]) -> NDArray[np.floating]:
    """Parallel transport on the SPD manifold with the affine-invariant metric.

    Transports tangent vector V at P to the tangent space at Q along
    the geodesic. Uses the Schild's ladder closed form:
    E = P^(1/2) (P^(-1/2) Q P^(-1/2))^(1/2) P^(-1/2), then V_t = E V E^T.

    Parameters
    ----------
    V : NDArray
        Tangent vector at P (symmetric matrix).
    P : NDArray
        Source point (n×n SPD matrix).
    Q : NDArray
        Destination point (n×n SPD matrix).

    Returns
    -------
    NDArray
        Transported tangent vector at Q (symmetric matrix).
    """
    V, P, Q = _f64(V), _f64(P), _f64(Q)
    P = _symm(P)
    Q = _symm(Q)
    V = _symm(V)
    P_sqrt = _sym_sqrt(P)
    P_inv_sqrt = _sym_inv_sqrt(P)
    E = P_sqrt @ _sym_sqrt(_symm(P_inv_sqrt @ Q @ P_inv_sqrt)) @ P_inv_sqrt
    return E @ V @ E.T
