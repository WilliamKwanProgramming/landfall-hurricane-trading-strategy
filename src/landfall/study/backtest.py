"""Event trading rules: market-hedged positions with equal risk, costs and borrow.

A trade in asset i opened at the close of trading day e and closed at the close of day e + H
earns, per unit of capital,

    gross = w s sum_{t = e+1}^{e+H} (R_it - beta_i R_mt),

with s = +1 (long the stock, short beta times the index) or -1 (the reverse), beta_i and the
residual sd sigma_i from the market model on the `est_len` days ending `gap` days before e (no
data after e is used), and w = min(target_vol / sigma_i, max_weight): every trade has the same
ex-ante daily residual risk (equal risk sizing). The notional is held constant (daily rebalanced)
so returns add. Costs (stated assumptions, `Costs`): `stock_bps` per side on the stock notional
w and `hedge_bps` per side on the index notional |beta| w, charged on entry and on exit (two
sides each); a short stock position pays `borrow_bps_per_year` on w for H / 252 years. The index
hedge is assumed to be a futures position with no borrow cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class Costs:
    stock_bps: float = 10.0
    hedge_bps: float = 2.0
    borrow_bps_per_year: float = 50.0


@dataclass(frozen=True)
class Sizing:
    target_vol: float = 0.01
    max_weight: float = 3.0
    est_len: int = 250
    gap: int = 30


def trade_returns(returns: NDArray, market: NDArray, entry_rows: NDArray, asset_cols: NDArray,
                  signs: NDArray, hold: int, costs: Costs | None = None,
                  sizing: Sizing | None = None) -> pd.DataFrame:
    """Gross and net return of each trade (rows of the output follow the inputs)."""
    costs = costs if costs is not None else Costs()
    sizing = sizing if sizing is not None else Sizing()
    out = []
    for e, a, s in zip(np.asarray(entry_rows), np.asarray(asset_cols), np.asarray(signs),
                       strict=True):
        lo = e - sizing.gap - sizing.est_len
        if lo < 0 or e + hold >= len(market):
            out.append((e, a, s, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        r_est = returns[lo: e - sizing.gap, a]
        m_est = market[lo: e - sizing.gap]
        xm = m_est - m_est.mean()
        beta = float((xm * (r_est - r_est.mean())).sum() / (xm ** 2).sum())
        alpha = r_est.mean() - beta * m_est.mean()
        sigma = float(np.sqrt(((r_est - alpha - beta * m_est) ** 2).sum() / (len(r_est) - 2)))
        w = min(sizing.target_vol / sigma, sizing.max_weight)
        hedged = returns[e + 1: e + hold + 1, a] - beta * market[e + 1: e + hold + 1]
        gross = w * s * float(hedged.sum())
        trading = 2.0 * (costs.stock_bps * w + costs.hedge_bps * abs(beta) * w) / 1e4
        borrow = (costs.borrow_bps_per_year / 1e4) * w * hold / 252.0 if s < 0 else 0.0
        out.append((e, a, s, w, beta, gross, trading + borrow, gross - trading - borrow))
    return pd.DataFrame(out, columns=["entry_row", "asset_col", "sign", "weight", "beta",
                                      "gross", "cost", "net"])


def non_overlapping(entry_rows: NDArray, asset_cols: NDArray, hold: int) -> NDArray:
    """Mask that keeps a trade only if no earlier kept trade in the same asset is still open
    (one position per asset at a time; inputs in time order). A trade may open at the close at
    which the previous one closes."""
    keep = np.zeros(len(entry_rows), dtype=bool)
    open_until: dict[int, int] = {}
    for k, (e, a) in enumerate(zip(entry_rows, asset_cols, strict=True)):
        if e >= open_until.get(int(a), -1):
            keep[k] = True
            open_until[int(a)] = int(e) + hold
    return keep


def leave_one_season_out(results: dict[str, pd.DataFrame], season_col: str = "season",
                         value: str = "net") -> pd.DataFrame:
    """Choose a configuration on the other seasons, evaluate it on the held-out season.

    `results` maps a configuration name to its trade table (with a season column and the
    `value` column), in a fixed order. For each season, the configuration with the highest mean
    `value` over the trades of all other seasons is chosen (ties: earliest in the order) and its
    trades of the held-out season are returned. Output: the held-out trades with the chosen
    configuration name.
    """
    names = list(results)
    seasons = sorted(set().union(*[set(df[season_col]) for df in results.values()]))
    parts = []
    for s in seasons:
        scores = []
        for n in names:
            other = results[n][results[n][season_col] != s][value].dropna()
            scores.append(other.mean() if len(other) else -np.inf)
        best = names[int(np.argmax(scores))]
        held = results[best][results[best][season_col] == s].copy()
        held["config"] = best
        parts.append(held)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
