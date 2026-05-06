"""Internal helpers: push/pull tangent vectors between Poincaré and hyperboloid.

The Poincaré ball ``D^n`` and the upper sheet of the Lorentz hyperboloid
``H^n`` are different coordinate charts of the same K=−1 hyperbolic
geometry. Tangent vectors at a Poincaré point ``p`` (which live in
``R^n``) and tangent vectors at the corresponding hyperboloid point
``phi(p)`` (which live in ``T_{phi(p)} H^n ⊂ R^{n+1}``) are related by
the Jacobian of the ``from_poincare`` map.

These two helpers are the linear maps:

  * ``push_poincare_tangent(v_poin, p_poin) -> v_hyp``
  * ``pull_poincare_tangent(v_hyp, p_poin) -> v_poin``

They are mutual inverses on the tangent space at ``phi(p)`` and preserve
the Riemannian / Minkowski inner product (the metric).

Used by both ``ophanimus.hyperbolic`` (the engine/dashboard module) and
``ophanimus.manifolds.poincare.parallel_transport`` (which routes through
the hyperboloid for the exact gyration).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._dtype import _f64


def push_poincare_tangent(v_poin: NDArray[np.floating], p_poin: NDArray[np.floating]) -> NDArray[np.floating]:
    """Push a Poincaré tangent at p to a hyperboloid tangent at from_poincare(p).

    The result is Minkowski-orthogonal to from_poincare(p) by construction.
    Inverse of ``pull_poincare_tangent`` on the tangent space at p.
    """
    v_poin = _f64(v_poin)
    p_poin = _f64(p_poin)
    s = 1.0 - np.dot(p_poin, p_poin)
    pv = np.dot(p_poin, v_poin)
    v0 = 4.0 * pv / (s * s)
    out = np.empty(p_poin.shape[0] + 1, dtype=np.float64)
    out[0] = v0
    out[1:] = 2.0 * v_poin / s + v0 * p_poin
    return out


def pull_poincare_tangent(v_hyp: NDArray[np.floating], p_poin: NDArray[np.floating]) -> NDArray[np.floating]:
    """Pull a hyperboloid tangent at from_poincare(p) back to a Poincaré tangent at p."""
    v_hyp = _f64(v_hyp)
    p_poin = _f64(p_poin)
    s = 1.0 - np.dot(p_poin, p_poin)
    return (s / 2.0) * (v_hyp[1:] - v_hyp[0] * p_poin)
