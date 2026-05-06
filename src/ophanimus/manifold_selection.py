import numpy as np
from numpy.typing import NDArray
from itertools import combinations

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from ._dtype import _f64
from .manifolds import poincare, sphere


# ---------------------------------------------------------------------------
# Distance matrix construction
# ---------------------------------------------------------------------------

def distance_from_features(X: NDArray[np.floating], metric: str = "euclidean") -> NDArray[np.floating]:
    """Build a pairwise distance matrix from feature vectors.

    Parameters
    ----------
    X : NDArray
        Feature matrix, shape (N, d). Each row is a data point.
    metric : str
        One of:
        - "euclidean": L2 distance.
        - "cosine": 1 - cosine_similarity. Ranges [0, 2].
        - "correlation": 1 - Pearson correlation. Ranges [0, 2].

    Returns
    -------
    NDArray
        Symmetric distance matrix, shape (N, N).
    """
    X = _f64(X)
    if metric not in {"euclidean", "cosine", "correlation"}:
        raise ValueError(
            f"Unknown metric: {metric}. Use 'euclidean', 'cosine', or 'correlation'."
        )
    return squareform(pdist(X, metric=metric))


def distance_from_similarity(S: NDArray[np.floating], method: str = "subtract") -> NDArray[np.floating]:
    """Convert a similarity matrix to a distance matrix.

    Parameters
    ----------
    S : NDArray
        Similarity matrix, shape (N, N). Higher = more similar.
    method : str
        Conversion method:
        - "subtract": d = max(S) - S. Simple, preserves ordering.
        - "inverse": d = 1/S. For strictly positive similarities.
        - "neglog": d = -log(S). For similarities in (0, 1].
          Produces additive distances (useful for probabilities).

    Returns
    -------
    NDArray
        Symmetric distance matrix, shape (N, N).
    """
    S = _f64(S)
    if method == "subtract":
        D = np.max(S) - S
    elif method == "inverse":
        D = 1.0 / np.maximum(S, 1e-15)
    elif method == "neglog":
        D = -np.log(np.maximum(S, 1e-15))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'subtract', 'inverse', or 'neglog'.")

    np.fill_diagonal(D, 0.0)
    return (D + D.T) / 2.0  # ensure symmetry


def distance_from_graph(adjacency: NDArray, weighted: bool = False) -> NDArray[np.floating]:
    """Build a shortest-path distance matrix from a graph adjacency matrix.

    Uses BFS for unweighted graphs and Dijkstra for weighted graphs.

    Parameters
    ----------
    adjacency : NDArray
        Adjacency matrix, shape (N, N).
        For unweighted: nonzero entries indicate edges.
        For weighted: entry values are edge weights (0 = no edge).
    weighted : bool
        If True, treat nonzero entries as edge weights (Dijkstra).
        If False, treat all edges as weight 1 (BFS).

    Returns
    -------
    NDArray
        Shortest-path distance matrix, shape (N, N).
        Unreachable pairs get distance = inf.
    """
    if weighted:
        graph = csr_matrix(_f64(adjacency))
        D = shortest_path(graph, directed=False, unweighted=False)
    else:
        # Treat any nonzero entry as a single-hop edge.
        graph = csr_matrix((np.asarray(adjacency) != 0).astype(np.int8))
        D = shortest_path(graph, directed=False, unweighted=True)
    return _f64(D)


# ---------------------------------------------------------------------------
# Stage 1: Structural test (no embedding needed)
# ---------------------------------------------------------------------------

def gromov_delta(D: NDArray[np.floating], n_samples: int = 500, seed: int = 42) -> dict:
    """Compute Gromov's delta-hyperbolicity of a distance matrix.

    Tests how "tree-like" the data is by examining quadruples of points.
    For each quadruple (a, b, c, d), compute the three sums of opposite
    pair distances, sort them, and measure the gap between the two largest.
    In a perfect tree, delta = 0. Small delta/diameter means hyperbolic
    geometry will fit well.

    Parameters
    ----------
    D : NDArray
        Pairwise distance matrix, shape (N, N). Must be symmetric.
    n_samples : int
        Number of random quadruples to sample. Use None for exhaustive
        computation (O(N^4), only feasible for small N).
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    dict with keys:
        delta : float
            The Gromov delta (max over sampled quadruples).
        delta_relative : float
            delta / diameter, normalized to [0, ~0.5].
            < 0.1 suggests strongly tree-like (hyperbolic).
            > 0.25 suggests not hierarchical.
        diameter : float
            Max pairwise distance in D.
        deltas : NDArray
            All individual delta values for the sampled quadruples.
    """
    D = _f64(D)
    N = D.shape[0]
    diameter = np.max(D)

    if diameter < 1e-15:
        return {"delta": 0.0, "delta_relative": 0.0, "diameter": 0.0,
                "deltas": np.array([0.0])}

    indices = np.arange(N)
    rng = np.random.default_rng(seed)

    # Generate quadruples
    if n_samples is None or N <= 30:
        quads = list(combinations(indices, 4))
    else:
        quads = [rng.choice(indices, size=4, replace=False) for _ in range(n_samples)]

    deltas = np.empty(len(quads))
    for idx, (a, b, c, d) in enumerate(quads):
        # Three ways to pair four points into two pairs
        s1 = D[a, b] + D[c, d]
        s2 = D[a, c] + D[b, d]
        s3 = D[a, d] + D[b, c]

        # Sort descending, delta is half the gap between the two largest
        sums = sorted([s1, s2, s3], reverse=True)
        deltas[idx] = (sums[0] - sums[1]) / 2.0

    delta = float(np.max(deltas))

    return {
        "delta": delta,
        "delta_relative": delta / diameter,
        "diameter": float(diameter),
        "deltas": deltas,
    }


# ---------------------------------------------------------------------------
# Stage 2: Embed and measure distortion
# ---------------------------------------------------------------------------

def _embed_euclidean(D: NDArray[np.floating], dim: int, steps: int = 1000,
                     lr: float = 0.01, seed: int = 42) -> NDArray[np.floating]:
    """Embed a distance matrix into Euclidean R^dim via stress minimization."""
    N = D.shape[0]
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.01, size=(N, dim))

    # Pairs and target distances
    ii, jj = np.triu_indices(N, k=1)
    d_target = D[ii, jj]
    mask = d_target > 1e-15

    for _ in range(steps):
        diffs = X[ii] - X[jj]                           # (P, dim)
        d_embed = np.linalg.norm(diffs, axis=1)          # (P,)
        d_embed_safe = np.maximum(d_embed, 1e-15)

        residuals = d_embed - d_target                   # (P,)
        # Gradient: push/pull each pair proportional to residual
        grad_scale = residuals / d_embed_safe             # (P,)
        grad_scale[~mask] = 0

        grad = grad_scale[:, None] * diffs                # (P, dim)

        # Accumulate gradients per point
        G = np.zeros_like(X)
        np.add.at(G, ii, grad)
        np.add.at(G, jj, -grad)

        X -= lr * G / N

    return X


def _embed_sphere(D: NDArray[np.floating], dim: int, steps: int = 1000,
                  lr: float = 0.01, seed: int = 42) -> NDArray[np.floating]:
    """Embed a distance matrix onto S^dim (ambient dim+1) via stress minimization."""
    N = D.shape[0]
    ambient = dim + 1
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(N, ambient))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    ii, jj = np.triu_indices(N, k=1)
    d_target = D[ii, jj]
    # Normalize target distances to [0, pi] range for the sphere
    d_max = np.max(d_target)
    if d_max > 1e-15:
        d_target_scaled = d_target * (np.pi / d_max)
    else:
        d_target_scaled = d_target

    for _ in range(steps):
        # Compute sphere distances
        dots = np.sum(X[ii] * X[jj], axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        d_embed = np.arccos(dots)

        residuals = d_embed - d_target_scaled

        # Gradient of arccos(dot(xi, xj)) w.r.t. xi is -xj / sin(angle)
        sin_d = np.sin(d_embed)
        sin_d_safe = np.maximum(sin_d, 1e-15)
        scale = -residuals / sin_d_safe

        grad_i = scale[:, None] * X[jj]  # gradient w.r.t. X[ii]
        grad_j = scale[:, None] * X[ii]  # gradient w.r.t. X[jj]

        G = np.zeros_like(X)
        np.add.at(G, ii, grad_i)
        np.add.at(G, jj, grad_j)

        # Project gradient onto tangent space (remove radial component)
        dots_gx = np.sum(G * X, axis=1, keepdims=True)
        G = G - dots_gx * X

        X -= lr * G / N
        # Re-project onto sphere
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

    return X


def _embed_poincare(D: NDArray[np.floating], dim: int, steps: int = 1000,
                    lr: float = 0.001, seed: int = 42) -> NDArray[np.floating]:
    """Embed a distance matrix into the Poincaré ball D^dim via stress minimization."""
    N = D.shape[0]
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.01, size=(N, dim))

    ii, jj = np.triu_indices(N, k=1)
    d_target = D[ii, jj]
    mask = d_target > 1e-15

    for step in range(steps):
        # Compute Poincaré distances
        diff_sq = np.sum((X[ii] - X[jj]) ** 2, axis=1)
        norm_i_sq = np.sum(X[ii] ** 2, axis=1)
        norm_j_sq = np.sum(X[jj] ** 2, axis=1)
        denom = (1 - norm_i_sq) * (1 - norm_j_sq)
        denom = np.maximum(denom, 1e-15)
        arg = 1 + 2 * diff_sq / denom
        arg = np.maximum(arg, 1.0 + 1e-15)
        d_embed = np.arccosh(arg)

        residuals = d_embed - d_target

        # Euclidean gradient of arccosh(1 + 2||xi-xj||^2 / ((1-||xi||^2)(1-||xj||^2)))
        # w.r.t. xi, then scale to Riemannian gradient by dividing by lambda^2
        cosh_d = arg
        sinh_d = np.sqrt(np.maximum(cosh_d ** 2 - 1, 1e-15))

        diff_vec = X[ii] - X[jj]  # (P, dim)

        # d(arccosh(f))/dx = (df/dx) / sqrt(f^2 - 1)
        # df/dxi = 4 / denom * (diff + xi * (2||xi-xj||^2 / ((1-||xi||^2)^2 * (1-||xj||^2))))
        # Simplified: use the chain rule through the Riemannian gradient
        alpha_i = (1 - norm_i_sq) ** 2 / 4.0  # 1/lambda_i^2
        alpha_j = (1 - norm_j_sq) ** 2 / 4.0

        grad_scale = residuals / np.maximum(sinh_d, 1e-15)
        grad_scale[~mask] = 0

        # Approximate Riemannian gradient via finite-length tangent direction
        # For each pair, the gradient pushes xi toward/away from xj
        d_embed_safe = np.maximum(d_embed, 1e-15)
        direction = diff_vec / np.linalg.norm(diff_vec, axis=1, keepdims=True).clip(1e-15)

        grad_i = (grad_scale * alpha_i)[:, None] * direction
        grad_j = -(grad_scale * alpha_j)[:, None] * direction

        G = np.zeros_like(X)
        np.add.at(G, ii, grad_i)
        np.add.at(G, jj, grad_j)

        X -= lr * G / N

        # Project back into the ball (norm < 1)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        too_big = norms > 0.95
        X = np.where(too_big, X * 0.95 / norms, X)

    return X


def distortion_ratios(D_original: NDArray[np.floating],
                      D_embedded: NDArray[np.floating]) -> dict:
    """Compute per-pair distortion ratios between original and embedded distances.

    Parameters
    ----------
    D_original : NDArray
        Original pairwise distance matrix (N, N).
    D_embedded : NDArray
        Embedded pairwise distance matrix (N, N).

    Returns
    -------
    dict with keys:
        ratios : NDArray
            Per-pair d_embedded / d_original for all upper-triangle pairs
            where d_original > 0.
        max_ratio : float
            Worst-case expansion (multiplicative distortion D).
        min_ratio : float
            Worst-case contraction.
        mean_ratio : float
            Average distortion ratio (1.0 = perfect).
        variance : float
            Variance of ratios. High = geometry warps some pairs more
            than others.
    """
    ii, jj = np.triu_indices(D_original.shape[0], k=1)
    d_orig = D_original[ii, jj]
    d_embed = D_embedded[ii, jj]

    mask = d_orig > 1e-15
    ratios = d_embed[mask] / d_orig[mask]

    return {
        "ratios": ratios,
        "max_ratio": float(np.max(ratios)) if len(ratios) > 0 else 0.0,
        "min_ratio": float(np.min(ratios)) if len(ratios) > 0 else 0.0,
        "mean_ratio": float(np.mean(ratios)) if len(ratios) > 0 else 0.0,
        "variance": float(np.var(ratios)) if len(ratios) > 0 else 0.0,
    }


def stress(D_original: NDArray[np.floating],
           D_embedded: NDArray[np.floating]) -> float:
    """Normalized stress between original and embedded distance matrices.

    stress = sqrt( sum((d_orig - d_embed)^2) / sum(d_orig^2) )

    Parameters
    ----------
    D_original : NDArray
        Original pairwise distance matrix (N, N).
    D_embedded : NDArray
        Embedded pairwise distance matrix (N, N).

    Returns
    -------
    float
        Normalized stress. 0 = perfect embedding, higher = more distortion.
    """
    ii, jj = np.triu_indices(D_original.shape[0], k=1)
    d_orig = D_original[ii, jj]
    d_embed = D_embedded[ii, jj]

    ss_res = np.sum((d_orig - d_embed) ** 2)
    ss_orig = np.sum(d_orig ** 2)
    if ss_orig < 1e-15:
        return 0.0
    return float(np.sqrt(ss_res / ss_orig))


def _pairwise_distances_euclidean(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """Pairwise Euclidean distance matrix from point array."""
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _pairwise_distances_sphere(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """Pairwise geodesic distance matrix for points on the sphere."""
    dots = X @ X.T
    dots = np.clip(dots, -1.0, 1.0)
    return np.arccos(dots)


def _pairwise_distances_poincare(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """Pairwise geodesic distance matrix for points in the Poincaré ball."""
    N = X.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            D[i, j] = poincare.distance(X[i], X[j])
            D[j, i] = D[i, j]
    return D


# ---------------------------------------------------------------------------
# Stage 2: Combined embed-and-measure
# ---------------------------------------------------------------------------

def embed_and_measure(D: NDArray[np.floating], geometry: str, dim: int,
                      steps: int = 1000, seed: int = 42) -> dict:
    """Embed a distance matrix into a candidate geometry and measure distortion.

    Parameters
    ----------
    D : NDArray
        Original pairwise distance matrix (N, N).
    geometry : str
        One of "euclidean", "sphere", "poincare".
    dim : int
        Embedding dimension (intrinsic dimension for sphere/poincare).
    steps : int
        Optimization steps for the embedding.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        geometry : str
        stress : float
        max_ratio : float (multiplicative distortion)
        min_ratio : float
        mean_ratio : float
        ratio_variance : float
        points : NDArray (the embedded points)
    """
    D = _f64(D)
    if geometry == "euclidean":
        X = _embed_euclidean(D, dim, steps=steps, seed=seed)
        D_embed = _pairwise_distances_euclidean(X)
    elif geometry == "sphere":
        X = _embed_sphere(D, dim, steps=steps, seed=seed)
        D_embed = _pairwise_distances_sphere(X)
    elif geometry == "poincare":
        X = _embed_poincare(D, dim, steps=steps, seed=seed)
        D_embed = _pairwise_distances_poincare(X)
    else:
        raise ValueError(f"Unknown geometry: {geometry}. Use 'euclidean', 'sphere', or 'poincare'.")

    # Normalize embedded distances to same scale as original for fair comparison
    ii, jj = np.triu_indices(D.shape[0], k=1)
    scale = np.sum(D[ii, jj]) / np.maximum(np.sum(D_embed[ii, jj]), 1e-15)
    D_embed_scaled = D_embed * scale

    s = stress(D, D_embed_scaled)
    dr = distortion_ratios(D, D_embed_scaled)

    return {
        "geometry": geometry,
        "stress": s,
        "max_ratio": dr["max_ratio"],
        "min_ratio": dr["min_ratio"],
        "mean_ratio": dr["mean_ratio"],
        "ratio_variance": dr["variance"],
        "points": X,
    }


# ---------------------------------------------------------------------------
# Full pipeline: select_manifold
# ---------------------------------------------------------------------------

def select_manifold(D: NDArray[np.floating], dim: int,
                    steps: int = 1000, seed: int = 42) -> dict:
    """Run the full manifold selection pipeline on a distance matrix.

    Stage 1: Compute Gromov delta-hyperbolicity (structural test).
    Stage 2: Embed into Euclidean, Sphere, and Poincaré at the given
             dimension and compare distortion.

    Parameters
    ----------
    D : NDArray
        Pairwise distance matrix (N, N). Symmetric, non-negative.
    dim : int
        Embedding dimension to use for all candidate geometries.
    steps : int
        Optimization steps for each embedding.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        gromov : dict
            Results from gromov_delta (delta, delta_relative, diameter).
        embeddings : dict
            Keyed by geometry name, each value is the embed_and_measure result.
        ranking : list of str
            Geometries ranked by stress (best first).
        recommendation : str
            The geometry with lowest stress.
    """
    D = _f64(D)
    # Stage 1: structural test
    gromov = gromov_delta(D)

    # Stage 2: competitive embedding
    geometries = ["euclidean", "sphere", "poincare"]
    results = {}
    for geom in geometries:
        results[geom] = embed_and_measure(D, geom, dim, steps=steps, seed=seed)

    # Rank by stress (lower = better fit)
    ranking = sorted(geometries, key=lambda g: results[g]["stress"])

    return {
        "gromov": {
            "delta": gromov["delta"],
            "delta_relative": gromov["delta_relative"],
            "diameter": gromov["diameter"],
        },
        "embeddings": results,
        "ranking": ranking,
        "recommendation": ranking[0],
    }
