import numpy as np
from numpy.typing import NDArray
from typing import Callable
from .frechet_mean import frechet_mean
from .._dtype import _f64


def pga(
    points: list[NDArray[np.floating]],
    exp_map: Callable,
    log_map: Callable,
    n_components: int = None,
) -> dict:
    """Principal Geodesic Analysis — manifold analogue of PCA.

    1. Compute the Fréchet mean of the data.
    2. Log-map all points to the tangent space at the mean.
    3. Run standard PCA in that tangent space.
    4. The principal components are tangent vectors at the mean
       representing the main directions of variation on the manifold.

    Parameters
    ----------
    points : list of NDArray
        Data points on the manifold.
    exp_map : callable
        Exponential map: (point, tangent) -> point.
    log_map : callable
        Logarithmic map: (point, point) -> tangent.
    n_components : int or None
        Number of principal components to return. If None, return all.

    Returns
    -------
    dict with keys:
        mean : NDArray
            Fréchet mean of the data.
        components : NDArray
            Principal tangent vectors at the mean, shape (n_components, d).
            Each row is a direction of variation on the manifold.
        explained_variance : NDArray
            Variance explained by each component.
        explained_variance_ratio : NDArray
            Fraction of total variance per component.
        tangent_coords : NDArray
            Data projected onto principal components, shape (N, n_components).
            These are the manifold-aware "scores."
    """
    points = [_f64(p) for p in points]
    N = len(points)
    mu = frechet_mean(points, exp_map, log_map)

    # Log-map everything to tangent space at the mean
    tangents = np.array([log_map(mu, points[i]) for i in range(N)])

    # Flatten if tangent vectors are matrices (e.g. SPD manifold)
    original_shape = tangents[0].shape
    if tangents[0].ndim > 1:
        tangents_flat = tangents.reshape(N, -1)
    else:
        tangents_flat = tangents

    # Standard PCA in tangent space
    centered = tangents_flat - tangents_flat.mean(axis=0)
    cov = (centered.T @ centered) / (N - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort by descending eigenvalue
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if n_components is not None:
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]

    total_var = np.sum(eigvals) if np.sum(eigvals) > 1e-15 else 1.0

    # Project data onto components
    coords = centered @ eigvecs

    # Reshape components back to tangent vector shape if needed
    if len(original_shape) > 1:
        components = eigvecs.T.reshape(-1, *original_shape)
    else:
        components = eigvecs.T

    return {
        "mean": mu,
        "components": components,
        "explained_variance": eigvals,
        "explained_variance_ratio": eigvals / total_var,
        "tangent_coords": coords,
    }
