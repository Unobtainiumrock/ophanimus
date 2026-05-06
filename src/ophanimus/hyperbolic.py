"""Engine/dashboard split: Poincaré-coordinated primitives that compute
through the Lorentz hyperboloid.

Why this module exists
----------------------
The Poincaré ball is the right *interpretive* coordinate system — bounded,
conformal, plottable. The Lorentz hyperboloid is the right *compute*
coordinate system — no boundary singularity, gradients well-conditioned
everywhere. The 2018 follow-up paper from Nickel and Kiela (the same
authors of the 2017 Poincaré paper) made this split explicit and
pivoted their own work to Lorentz.

The functions here are drop-in replacements for the corresponding
``ophanimus.manifolds.poincare`` primitives. Inputs and outputs are in
Poincaré ball coordinates so any algorithm that already accepts callable
``exp_map`` / ``log_map`` / ``distance`` / ``parallel_transport`` arguments
will Just Work::

    from ophanimus import hyperbolic
    from ophanimus.algorithms import kmeans

    labels, centers = kmeans(
        points,                   # Poincaré-coordinated
        k=3,
        exp_map=hyperbolic.exp_map,
        log_map=hyperbolic.log_map,
        distance=hyperbolic.distance,
    )

Internally each call:

  1. Lifts Poincaré inputs to the hyperboloid via ``from_poincare``
     (or via the tangent-space Jacobian for tangent vectors).
  2. Runs the hyperboloid implementation.
  3. Projects hyperboloid outputs back to the Poincaré ball.

The math
--------
The from_poincare map ``phi: D^n -> H^n`` is::

    phi(p)_0 = (1+|p|²) / (1-|p|²)
    phi(p)_i = 2 p_i / (1-|p|²),   i = 1..n

Tangent vectors transform via the Jacobian of phi. Let
``s = 1 - |p|²``. Then for a Poincaré tangent ``v_poin ∈ R^n``::

    v_hyp_0 = 4 (p · v_poin) / s²
    v_hyp_i = 2 v_poin_i / s + v_hyp_0 · p_i

This v_hyp satisfies ``<phi(p), v_hyp>_M = 0`` (Minkowski-orthogonal),
so it lives in T_{phi(p)} H^n.

The inverse (pulling a hyperboloid tangent at phi(p) back to a Poincaré
tangent at p) is::

    v_poin = (s / 2) · (v_hyp[1:] - v_hyp[0] · p)

``_push_tangent`` and ``_pull_tangent`` are mutual inverses on the
tangent space at p (verified by composition + Minkowski-orthogonality).

Curvature
---------
Only ``c = 1`` (K = -1) is supported here. The hyperboloid module is
written for the unit-curvature Lorentz model. If you need variable
curvature, use ``ophanimus.manifolds.poincare`` directly with the ``c``
parameter — those primitives handle ``c`` natively.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._dtype import _f64
from ._tangent import push_poincare_tangent as _push_tangent  # private alias for back-compat
from ._tangent import pull_poincare_tangent as _pull_tangent  # private alias for back-compat
from .manifolds import hyperboloid as _hyp


# ---------------------------------------------------------------------------
# Public primitives — Poincaré in, Poincaré out, hyperboloid in the middle
# ---------------------------------------------------------------------------

def distance(p: NDArray[np.floating], q: NDArray[np.floating]) -> np.floating:
    """Geodesic distance between two Poincaré points, computed via the hyperboloid.

    Numerically stable up to and well past ``||p|| = 0.999`` because the
    hyperboloid distance ``arccosh(-<u,v>_M)`` has no boundary singularity.
    """
    p, q = _f64(p), _f64(q)
    return _hyp.distance(_hyp.from_poincare(p), _hyp.from_poincare(q))


def exp_map(p: NDArray[np.floating], v: NDArray[np.floating]) -> NDArray[np.floating]:
    """Exponential map: ``exp_p(v)`` for a Poincaré base ``p`` and Poincaré tangent ``v``.

    Path: lift ``p`` and push ``v`` to the hyperboloid, run hyperboloid exp,
    project the result back to the ball.
    """
    p, v = _f64(p), _f64(v)
    p_hyp = _hyp.from_poincare(p)
    v_hyp = _push_tangent(v, p)
    q_hyp = _hyp.exp_map(p_hyp, v_hyp)
    return _hyp.to_poincare(q_hyp)


def log_map(p: NDArray[np.floating], q: NDArray[np.floating]) -> NDArray[np.floating]:
    """Logarithmic map: ``log_p(q)`` for two Poincaré points.

    Path: lift both to the hyperboloid, run hyperboloid log, pull the
    resulting tangent back to the Poincaré tangent space at ``p``.
    """
    p, q = _f64(p), _f64(q)
    p_hyp = _hyp.from_poincare(p)
    q_hyp = _hyp.from_poincare(q)
    v_hyp = _hyp.log_map(p_hyp, q_hyp)
    return _pull_tangent(v_hyp, p)


def parallel_transport(v: NDArray[np.floating], p: NDArray[np.floating], q: NDArray[np.floating]) -> NDArray[np.floating]:
    """Parallel transport a Poincaré tangent ``v`` from ``p`` to ``q``.

    Path: push ``v`` to the hyperboloid tangent at ``from_poincare(p)``,
    transport along the geodesic on the hyperboloid, pull the result back
    to the Poincaré tangent space at ``q``.
    """
    v, p, q = _f64(v), _f64(p), _f64(q)
    p_hyp = _hyp.from_poincare(p)
    q_hyp = _hyp.from_poincare(q)
    v_hyp_at_p = _push_tangent(v, p)
    v_hyp_at_q = _hyp.parallel_transport(v_hyp_at_p, p_hyp, q_hyp)
    return _pull_tangent(v_hyp_at_q, q)
