"""Stationary bootstrap (Politis and Romano 1994) for the mean of a time-ordered series.

Resamples blocks whose lengths are geometric with mean `mean_block` (each next observation
continues the current block with probability 1 - 1/mean_block, otherwise a new block starts at a
uniform random position), wrapping around the end of the series. mean_block = 1 is the i.i.d.
bootstrap. Percentile intervals.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def stationary_indices(n: int, mean_block: float, n_boot: int, rng: np.random.Generator
                       ) -> NDArray:
    """(n_boot, n) index matrix of stationary-bootstrap resamples of 0..n-1."""
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    p_new = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    new = rng.random((n_boot, n)) < p_new
    starts = rng.integers(0, n, (n_boot, n))
    for j in range(1, n):
        idx[:, j] = np.where(new[:, j], starts[:, j], (idx[:, j - 1] + 1) % n)
    return idx


def bootstrap_mean_ci(x: NDArray, mean_block: float, n_boot: int = 2000, seed: int = 0,
                      level: float = 0.95) -> tuple[float, float]:
    """Percentile interval for the mean of x from the stationary bootstrap."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[stationary_indices(x.size, mean_block, n_boot, rng)].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return float(lo), float(hi)
