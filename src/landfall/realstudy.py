"""Building blocks of the pre-specified real-data event study (docs/protocol.md).

Everything here works on a return matrix aligned with a trading calendar and on event times;
experiments/real_eventstudy.py wires it to the skfolio prices and the HURDAT2 events.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from landfall.study import bootstrap as bs
from landfall.study import eventstudy as es
from landfall.study import inference as inf

MAX_ZERO_SHARE = 0.10
OVERLAP_DAYS = 15
PLACEBO_EXCLUSION_DAYS = 21


def drop_overlapping(day0: NDArray, min_gap: int = OVERLAP_DAYS) -> NDArray:
    """Keep mask: events sorted by day 0 are dropped if day 0 is within `min_gap` trading days
    after the day 0 of the last kept event (inputs in any order; mask in input order)."""
    order = np.argsort(day0, kind="stable")
    keep = np.zeros(len(day0), dtype=bool)
    last = None
    for k in order:
        if last is None or day0[k] - last > min_gap:
            keep[k] = True
            last = day0[k]
    return keep


def quality_ok(ret: NDArray, day0: int, est_len: int = es.EST_LEN, gap: int = es.GAP,
               window: tuple[int, int] = es.WINDOW, max_zero: float = MAX_ZERO_SHARE) -> bool:
    """False if more than `max_zero` of the returns in the estimation window, or in the event
    window, are exactly zero (stale prices)."""
    est = ret[day0 - gap - est_len: day0 - gap]
    win = ret[day0 + window[0]: day0 + window[1] + 1]
    return bool((est == 0).mean() <= max_zero and (win == 0).mean() <= max_zero)


def window_fits(day0: int, n_days: int, est_len: int = es.EST_LEN, gap: int = es.GAP,
                window: tuple[int, int] = es.WINDOW) -> bool:
    return day0 >= 0 and day0 - gap - est_len >= 0 and day0 + window[1] < n_days


@dataclass
class AssetEvents:
    """Market-model results of one asset over its usable events."""

    asset: str
    events: NDArray  # indices into the event list
    day0: NDArray
    mm: es.MarketModel


def fit_asset(returns: NDArray, market: NDArray, day0: NDArray, asset: str) -> tuple[
        AssetEvents | None, int]:
    """Market model for every event of one asset that passes the data screen.

    Returns the fit (None if no event passes) and the number of events excluded by the screen.
    """
    ok = np.array([quality_ok(returns, int(d)) for d in day0])
    if not ok.any():
        return None, int((~ok).sum())
    d = day0[ok]
    mm = es.fit_market_model(*es.extract(returns, market, d))
    return AssetEvents(asset, np.nonzero(ok)[0], d, mm), int((~ok).sum())


def window_stats(car: NDArray, scar: NDArray, mm: es.MarketModel, t1: int, t2: int,
                 mean_block: float, seed: int, n_boot: int = 2000,
                 rows: NDArray | None = None) -> dict:
    """Mean CAR and all tests for one window (rows selects asset-events of `mm`)."""
    rows = np.arange(len(car)) if rows is None else rows
    cols = slice(mm.idx(t1), mm.idx(t2) + 1)
    lo, hi = bs.bootstrap_mean_ci(car, mean_block, n_boot, seed)
    return {
        "n": len(car), "mean_car": float(car.mean()), "median_car": float(np.median(car)),
        "boot_lo": lo, "boot_hi": hi,
        "bmp_t": inf.bmp(scar).stat, "bmp_p": inf.bmp(scar).p,
        "patell_z": inf.patell(scar, mm.t_est).stat, "patell_p": inf.patell(scar, mm.t_est).p,
        "naive_t": inf.naive_t(car).stat, "naive_p": inf.naive_t(car).p,
        "rank_z": inf.rank_test(mm.resid_est[rows], mm.ar[rows], cols).stat,
        "rank_p": inf.rank_test(mm.resid_est[rows], mm.ar[rows], cols).p,
    }


def placebo_day0(event_times: pd.Series, landfall_times: pd.Series, dates: pd.DatetimeIndex,
                 years: tuple[int, int], n_sets: int, rng: np.random.Generator,
                 exclusion_days: int = PLACEBO_EXCLUSION_DAYS, max_tries: int = 1000) -> NDArray:
    """(n_sets, n_events) placebo day-0 rows (see docs/protocol.md).

    For each event: a different year drawn uniformly from `years`, a day drawn uniformly from the
    seven days of the event's ISO calendar week in that year; redrawn if any landfall time lies
    within `exclusion_days` calendar days or if the windows do not fit the price data.
    """
    ev = pd.DatetimeIndex(pd.to_datetime(event_times))
    lf = np.sort(pd.DatetimeIndex(pd.to_datetime(landfall_times)).normalize().to_numpy())
    all_years = np.arange(years[0], years[1] + 1)
    out = np.full((n_sets, len(ev)), -1, dtype=np.int64)
    excl = np.timedelta64(exclusion_days, "D")
    # Closes of the calendar computed once; the row of a candidate is what es.event_rows gives.
    closes = (pd.DatetimeIndex(dates).normalize() + pd.Timedelta(hours=20)).to_numpy()
    for j, t in enumerate(ev):
        week = t.isocalendar().week
        for s in range(n_sets):
            for _ in range(max_tries):
                y = int(rng.choice(all_years[all_years != t.year]))
                try:
                    monday = pd.Timestamp.fromisocalendar(y, int(week), 1)
                except ValueError:  # week 53 does not exist in that year
                    continue
                cand = monday + pd.Timedelta(days=int(rng.integers(0, 7)))
                c64 = np.datetime64(cand.normalize())
                k = np.searchsorted(lf, c64)
                near = (k < len(lf) and lf[k] - c64 <= excl) or (k > 0 and c64 - lf[k - 1] <= excl)
                if near:
                    continue
                row = int(np.searchsorted(closes, np.datetime64(cand + pd.Timedelta(hours=12)),
                                          side="left"))
                if row < len(closes) and window_fits(row, len(dates)):
                    out[s, j] = row
                    break
            else:
                raise RuntimeError(f"no placebo date found for event {j}")
    return out


def placebo_position(real: float, placebo: NDArray) -> tuple[float, float]:
    """Fraction of placebo values <= real, and the two-sided placebo p-value."""
    le = float((placebo <= real).mean())
    ge = float((placebo >= real).mean())
    return le, min(1.0, 2 * min(le, ge))


def car_for_rows(returns: NDArray, market: NDArray, day0: NDArray, t1: int, t2: int
                 ) -> NDArray:
    """CAR(t1, t2) for one asset at the given day-0 rows (no data screen; placebo use)."""
    mm = es.fit_market_model(*es.extract(returns, market, day0))
    return mm.car(t1, t2)


def entry_rows_from_times(dates: pd.DatetimeIndex, times: pd.Series,
                          close_utc_hour: float = 20.0) -> NDArray:
    """Row of the first close at or after each time (the entry close of a signal)."""
    return es.event_rows(dates, times, close_utc_hour)
