"""Multiple-testing adjustments of p-values.

Holm (1979): sort p ascending, adjusted p_(k) = max_{j <= k} min(1, (m - j + 1) p_(j)).
Benjamini and Hochberg (1995): adjusted p_(k) = min_{j >= k} min(1, m p_(j) / j).
Both return adjusted p-values in the original order; reject where adjusted p <= alpha.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def holm(p: ArrayLike) -> NDArray:
    p = np.asarray(p, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
    adj = np.maximum.accumulate(adj)
    out = np.empty(m)
    out[order] = adj
    return out


def benjamini_hochberg(p: ArrayLike) -> NDArray:
    p = np.asarray(p, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.minimum(1.0, m * p[order] / np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = adj
    return out
