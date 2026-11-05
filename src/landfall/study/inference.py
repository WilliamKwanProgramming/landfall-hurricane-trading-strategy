"""Cross-sectional tests of mean (cumulative) abnormal returns.

All statistics are for H0: mean CAR = 0 over N asset-events; two-sided p-values.

- Naive cross-sectional t: t = mean(CAR) / (sd(CAR) / sqrt(N)), t(N - 1). Ignores
  cross-sectional correlation.
- Patell (1976) Z: Z = sum SCAR_i / sqrt( sum (T_i - 2) / (T_i - 4) ), N(0, 1). Uses the
  estimation-period variance, so event-induced variance makes it over-reject.
- Boehmer, Musumeci and Poulsen (1991) standardised cross-sectional t:
  t = mean(SCAR) / (sd(SCAR) / sqrt(N)), t(N - 1) reference distribution.
- Kolari and Pynnonen (2010) adjustment for cross-sectional correlation of abnormal returns:
  t_adj = t sqrt( (1 - rbar) / (1 + (N - 1) rbar) ), rbar the average pairwise correlation of
  estimation-period residuals. Asset-events on different event dates have disjoint windows and are
  treated as uncorrelated (their r = 0 enters the average); pairs that share an event date use
  their sample correlation.
- Corrado (1989) rank test with the Cowan (1992) extension to multi-day windows. Ranks K_it of
  AR_it over the L = estimation + event days of each asset-event (average ranks for ties);
  U_t = mean_i (K_it - (L + 1) / 2); S = sqrt( (1/L) sum_t U_t^2 );
  Z = sum_{t in window} U_t / (sqrt(D) S), N(0, 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    stat: float
    p: float
    n: int


def _two_sided_t(t: float, df: int) -> float:
    return float(2 * stats.t.sf(abs(t), df))


def _two_sided_z(z: float) -> float:
    return float(2 * stats.norm.sf(abs(z)))


def naive_t(car: NDArray) -> TestResult:
    n = car.size
    t = car.mean() / (car.std(ddof=1) / np.sqrt(n))
    return TestResult(float(t), _two_sided_t(t, n - 1), n)


def patell(scar: NDArray, t_est: NDArray | int) -> TestResult:
    n = scar.size
    t_est = np.broadcast_to(np.asarray(t_est, dtype=float), scar.shape)
    z = scar.sum() / np.sqrt(((t_est - 2) / (t_est - 4)).sum())
    return TestResult(float(z), _two_sided_z(z), n)


def bmp(scar: NDArray) -> TestResult:
    n = scar.size
    t = scar.mean() / (scar.std(ddof=1) / np.sqrt(n))
    return TestResult(float(t), _two_sided_t(t, n - 1), n)


def kp_factor(rbar: float, n: int) -> float:
    """Kolari-Pynnonen multiplier sqrt((1 - rbar) / (1 + (n - 1) rbar))."""
    return float(np.sqrt((1 - rbar) / (1 + (n - 1) * rbar)))


def bmp_kp(scar: NDArray, rbar: float) -> TestResult:
    b = bmp(scar)
    t = b.stat * kp_factor(rbar, b.n)
    return TestResult(float(t), _two_sided_t(t, b.n - 1), b.n)


def average_cross_correlation(resid_est: NDArray, groups: NDArray) -> float:
    """Average pairwise residual correlation over all N(N-1)/2 pairs; pairs in different
    groups (event dates) count as zero."""
    n = resid_est.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    for g in np.unique(groups):
        idx = np.nonzero(groups == g)[0]
        if len(idx) < 2:
            continue
        c = np.corrcoef(resid_est[idx])
        total += (c.sum() - np.trace(c)) / 2.0
    return float(total / (n * (n - 1) / 2.0))


def rank_test(est_ar: NDArray, win_ar: NDArray, cols: slice | NDArray) -> TestResult:
    """Corrado (1989) / Cowan (1992) rank test for the event-window columns `cols` of win_ar."""
    full = np.concatenate([est_ar, win_ar], axis=1)
    n, length = full.shape
    ranks = stats.rankdata(full, axis=1)
    u = (ranks - (length + 1) / 2.0).mean(axis=0)  # (L,)
    s = np.sqrt((u ** 2).mean())
    uw = u[est_ar.shape[1]:][cols]
    d = uw.size
    z = uw.sum() / (np.sqrt(d) * s)
    return TestResult(float(z), _two_sided_z(z), n)


def t_interval(x: NDArray, level: float = 0.95) -> tuple[float, float]:
    """Student-t interval for the mean of x (analytic)."""
    n = x.size
    h = stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean() - h), float(x.mean() + h)
