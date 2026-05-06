import numpy as np
from numpy.typing import NDArray
from typing import Callable
from .frechet_mean import frechet_mean
from .._dtype import _f64


def geodesic_regression(
    X: NDArray[np.floating],
    Y: list[NDArray[np.floating]],
    exp_map: Callable,
    log_map: Callable,
    distance: Callable,
    parallel_transport: Callable,
    alpha_init: float = 1.0,
    alpha_max: float = 100.0,
    steps: int = 1000,
    tol: float = 1e-8,
) -> tuple[NDArray[np.floating], NDArray[np.floating], list[float]]:
    """Geodesic regression via Riemannian gradient descent.

    Fits a geodesic curve y_hat = Exp_p(x * v) to manifold-valued data,
    where p is a base point on the manifold and v is a tangent vector at p.

    This is the manifold analogue of linear regression y = p + x * v.

    Uses parallel transport to maintain geometric correctness of v when
    p moves, and adaptive line search to auto-tune the step size.

    Parameters
    ----------
    X : NDArray
        Scalar predictors, shape (N,).
    Y : list of NDArray
        Manifold-valued responses, one per data point.
    exp_map : callable
        Exponential map: (point, tangent) -> point.
    log_map : callable
        Logarithmic map: (point, point) -> tangent.
    distance : callable
        Geodesic distance: (point, point) -> scalar.
    parallel_transport : callable
        Parallel transport: (tangent, source_point, dest_point) -> tangent.
    alpha_init : float
        Initial step size for line search.
    alpha_max : float
        Maximum step size the line search can grow to.
    steps : int
        Maximum number of iterations.
    tol : float
        Stop early if loss changes less than this between iterations.

    Returns
    -------
    p : NDArray
        Fitted base point on the manifold.
    v : NDArray
        Fitted tangent vector at p (the "slope").
    losses : list of float
        Loss at each iteration for diagnostics.
    """
    X = _f64(X)
    Y = [_f64(y) for y in Y]
    N = len(Y)
    alpha = alpha_init

    # Initialize p at the Fréchet mean of Y
    p = frechet_mean(Y, exp_map, log_map)

    # Initialize v via tangent-space least squares
    tangents = [log_map(p, Y[i]) for i in range(N)]
    W = np.array(tangents)
    x_sq_sum = np.dot(X, X)
    if x_sq_sum > 1e-15:
        v = np.tensordot(X, W, axes=([0], [0])) / x_sq_sum
    else:
        v = np.zeros_like(p)

    # Loss function
    def compute_loss(p_, v_):
        return 0.5 * np.sum([distance(Y[i], exp_map(p_, X[i] * v_)) ** 2 for i in range(N)])

    losses = []
    current_loss = compute_loss(p, v)

    # Riemannian gradient descent with line search
    for step in range(steps):
        losses.append(float(current_loss))

        # Gradients via tangent-space linearization
        errors = []
        for i in range(N):
            w_i = log_map(p, Y[i])
            v_hat_i = X[i] * v
            errors.append(w_i - v_hat_i)

        grad_p = -np.mean(errors, axis=0)
        grad_v = -np.mean([X[i] * errors[i] for i in range(N)], axis=0)

        # Line search: try the step, accept or shrink
        p_new = exp_map(p, -alpha * grad_p)

        # Transport v and grad_v to the new tangent space
        v_transported = parallel_transport(v, p, p_new)
        v_new = v_transported - alpha * parallel_transport(grad_v, p, p_new)

        new_loss = compute_loss(p_new, v_new)

        if new_loss < current_loss:
            p = p_new
            v = v_new
            if abs(current_loss - new_loss) < tol:
                losses.append(float(new_loss))
                break
            current_loss = new_loss
            alpha = min(2 * alpha, alpha_max)
        else:
            alpha = alpha / 2.0
            if alpha < 1e-12:
                break

    return p, v, losses
