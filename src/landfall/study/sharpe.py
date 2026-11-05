"""Sharpe ratio, probabilistic Sharpe ratio and deflated Sharpe ratio.

SR = mean / sd (sd with ddof = 1) of a return series, per period; annualised SR = SR sqrt(k)
for k periods per year.

Probabilistic Sharpe ratio (Bailey and Lopez de Prado 2012, "The Sharpe ratio efficient
frontier", Journal of Risk 15(2)):
    PSR(SR*) = Phi( (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1) / 4 SR^2) ),
with per-period SR, sample skewness g3 and sample (non-excess) kurtosis g4 of the T returns.

Deflated Sharpe ratio (Bailey and Lopez de Prado 2014, "The Deflated Sharpe Ratio: Correcting
for Selection Bias, Backtest Overfitting and Non-Normality", Journal of Portfolio Management
40(5)): DSR = PSR(SR0) with the expected maximum of N independent trials' Sharpe ratios
    SR0 = sqrt(V) [ (1 - gamma) Phi^{-1}(1 - 1/N) + gamma Phi^{-1}(1 - 1/(N e)) ],
V the variance of the trials' Sharpe ratios and gamma the Euler-Mascheroni constant. The
implementation reproduces the numbers of the paper's worked example (tests/test_sharpe.py).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def sharpe(x: ArrayLike) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.mean() / x.std(ddof=1))


def annualise(sr: float, periods_per_year: float) -> float:
    return float(sr * np.sqrt(periods_per_year))


def moments(x: ArrayLike) -> tuple[float, float]:
    """Sample skewness and (non-excess) kurtosis, biased estimators as in the papers."""
    x = np.asarray(x, dtype=float)
    return float(stats.skew(x, bias=True)), float(stats.kurtosis(x, fisher=False, bias=True))


def psr(sr: float, t: int, skew: float, kurt: float, sr_star: float = 0.0) -> float:
    den = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(t - 1) / den))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def dsr(sr: float, t: int, skew: float, kurt: float, n_trials: int, var_sr: float) -> float:
    return psr(sr, t, skew, kurt, expected_max_sharpe(n_trials, var_sr))
