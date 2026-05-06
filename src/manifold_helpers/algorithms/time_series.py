import numpy as np
from numpy.typing import NDArray
from typing import Callable

from .._dtype import _f64


def manifold_velocities(
    trajectory: list[NDArray[np.floating]],
    log_map: Callable,
) -> list[NDArray[np.floating]]:
    """Compute tangent velocities along a manifold-valued trajectory.

    Given a sequence of points on the manifold, computes the tangent
    vector from each point to the next: v_i = Log_{p_i}(p_{i+1}).

    Parameters
    ----------
    trajectory : list of NDArray
        Ordered sequence of points on the manifold.
    log_map : callable
        Logarithmic map: (point, point) -> tangent.

    Returns
    -------
    list of NDArray
        Tangent velocities, length len(trajectory) - 1.
        Each v_i lives in the tangent space at trajectory[i].
    """
    trajectory = [_f64(p) for p in trajectory]
    return [log_map(trajectory[i], trajectory[i + 1])
            for i in range(len(trajectory) - 1)]


def compare_velocities(
    trajectory: list[NDArray[np.floating]],
    log_map: Callable,
    parallel_transport: Callable,
    reference_index: int = 0,
) -> list[NDArray[np.floating]]:
    """Transport all velocities to a common tangent space for comparison.

    Raw velocities from manifold_velocities live in different tangent
    spaces (one per time step), so you can't directly compare or average
    them. This function parallel-transports all velocities to the tangent
    space at a reference point.

    Parameters
    ----------
    trajectory : list of NDArray
        Ordered sequence of points on the manifold.
    log_map : callable
        Logarithmic map: (point, point) -> tangent.
    parallel_transport : callable
        Parallel transport: (tangent, source, dest) -> tangent.
    reference_index : int
        Index of the reference point whose tangent space to transport to.

    Returns
    -------
    list of NDArray
        All velocities transported to the tangent space at
        trajectory[reference_index]. Can be directly compared,
        averaged, or analyzed (e.g. fed into PGA).
    """
    trajectory = [_f64(p) for p in trajectory]
    velocities = manifold_velocities(trajectory, log_map)
    ref = trajectory[reference_index]

    transported = []
    for i, v in enumerate(velocities):
        source = trajectory[i]
        transported.append(parallel_transport(v, source, ref))

    return transported
