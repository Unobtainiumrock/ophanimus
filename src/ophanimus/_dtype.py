"""Internal dtype-promotion helper.

Every public function in this package promotes its array inputs to
float64 at the boundary. Riemannian-gradient stability tricks (clip
near boundaries, eps guards, arctanh clipping) are tuned for fp64;
fp32 inputs would silently lose accuracy in those operations.

Scalars (Python int/float) stay native — only arrays are promoted.
"""
from __future__ import annotations

import numpy as np


def _f64(x):
    return np.asarray(x, dtype=np.float64)
