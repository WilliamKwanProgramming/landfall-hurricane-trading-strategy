"""Power of the per-asset event-study tests, by simulation at a chosen event count and volatility.

`power_curve` simulates `reps` independent panels of one asset with `n_events` events (scenario of
`landfall.study.simulate`, default GARCH(1,1)-t(5)), a residual sd `sigma` a day and a true
abnormal return `delta` added to CAR[0, +10], and reports the rejection rate at 5% of each test
(the size at delta = 0). `interpolate_target` finds the smallest delta at which a power curve
reaches a target; `plugin_t_power` is an analytic cross-check that needs no simulation.
Used by experiments/per_stock_power.py (the eight real stocks at their own volatility).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from landfall.study import simulate as sim

BATCH = 100


def power_curve(n_events: int, sigma: float, deltas: Sequence[float], reps: int,
                rng_key: Sequence[int], scenario: str = "garch_t") -> pd.DataFrame:
    """One row per delta: rejection rate at 5% of each test and its Monte Carlo SE.

    The random generator of delta number k is `numpy.random.default_rng([*rng_key, k])`; no
    bootstrap interval is drawn. Columns: delta, reps, reject_<test>, reject_<test>_mcse for the
    tests in `simulate.TESTS`.
    """
    sc = sim.SCENARIOS[scenario]
    rows = []
    for k, delta in enumerate(deltas):
        rng = np.random.default_rng([*rng_key, k])
        res: list[dict] = []
        while len(res) < reps:
            res += sim.replicate(sc, n_events, 1, delta, rng, n_reps=min(BATCH, reps - len(res)),
                                 n_boot=0, sigma_e=sigma)
        df = pd.DataFrame(res)
        row: dict = {"delta": float(delta), "reps": reps}
        for t in sim.TESTS:
            p = float((df[f"p_{t}"] < 0.05).mean())
            row[f"reject_{t}"] = p
            row[f"reject_{t}_mcse"] = float(np.sqrt(p * (1 - p) / reps))
        rows.append(row)
    return pd.DataFrame(rows)


def interpolate_target(deltas: Sequence[float], power: Sequence[float], target: float) -> float:
    """Smallest delta at which the power curve reaches `target`, by linear interpolation between
    grid points (the first crossing); NaN if it is not reached on the grid. `deltas` ascending."""
    d, p = np.asarray(deltas, dtype=float), np.asarray(power, dtype=float)
    if p[0] >= target:
        return float(d[0])
    for k in range(1, len(d)):
        if p[k] >= target > p[k - 1]:
            return float(d[k - 1] + (target - p[k - 1]) / (p[k] - p[k - 1]) * (d[k] - d[k - 1]))
    return float("nan")


def plugin_t_power(se: float, n: int, delta: float) -> float:
    """Power of a two-sided 5% t test of a zero mean (n - 1 degrees of freedom) when the true
    mean is `delta` and the standard error of the sample mean is `se` (noncentral t; the standard
    error is treated as known, so heavy tails and estimation error are ignored)."""
    nc = delta / se
    tc = stats.t.ppf(0.975, n - 1)
    return float(stats.nct.sf(tc, n - 1, nc) + stats.nct.cdf(-tc, n - 1, nc))
