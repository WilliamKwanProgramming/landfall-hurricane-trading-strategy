"""Market-model event study in event time, vectorised over asset-events.

Timing (pre-specified, see docs/protocol.md). Day 0 is the trading day whose close is the first
close at or after the landfall time. The event window is days -5 .. +10. The estimation window is
the `est_len` = 250 trading days ending `gap` = 30 trading days before day 0, i.e. days -280 .. -31,
so it never overlaps the event window.

Market model (OLS on the estimation window, T = est_len days):
    R_it = alpha_i + beta_i R_mt + e_it,   s_i^2 = sum e_it^2 / (T - 2),
abnormal return AR_it = R_it - alpha_i - beta_i R_mt for t in the event window, and
CAR_i(t1, t2) = sum_{t = t1..t2} AR_it.

Variance of a CAR over D days, including the estimation error of alpha and beta (exact for OLS
with fixed market returns; it reduces to the one-day forecast-error variance of Patell (1976)
and Boehmer, Musumeci and Poulsen (1991) when D = 1):
    Var(CAR_i) = s_i^2 [ D + D^2 / T + (sum_{t in window} (R_mt - Rbar_m))^2 / SSx ],
with Rbar_m and SSx = sum (R_mt - Rbar_m)^2 over the estimation window. The standardised CAR is
SCAR_i = CAR_i / sqrt(Var(CAR_i)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

WINDOW = (-5, 10)
EST_LEN = 250
GAP = 30
CAR_WINDOWS = {"pre[-5,-1]": (-5, -1), "post[0,+10]": (0, 10), "full[-5,+10]": (-5, 10)}


@dataclass
class MarketModel:
    """Fitted market models and abnormal returns for M asset-events."""

    alpha: NDArray  # (M,)
    beta: NDArray  # (M,)
    sigma: NDArray  # (M,) residual sd with T - 2 degrees of freedom
    t_est: int
    mbar: NDArray  # (M,) mean market return in the estimation window
    ssx: NDArray  # (M,)
    resid_est: NDArray  # (M, T) estimation-window residuals
    ar: NDArray  # (M, W) abnormal returns in the event window
    win_m: NDArray  # (M, W) market returns in the event window
    window: tuple[int, int]

    def idx(self, day: int) -> int:
        """Column of event day `day` in `ar`."""
        if not self.window[0] <= day <= self.window[1]:
            raise ValueError(f"day {day} outside the event window {self.window}")
        return day - self.window[0]

    def car(self, t1: int, t2: int) -> NDArray:
        return self.ar[:, self.idx(t1): self.idx(t2) + 1].sum(axis=1)

    def car_variance(self, t1: int, t2: int) -> NDArray:
        d = t2 - t1 + 1
        dm = (self.win_m[:, self.idx(t1): self.idx(t2) + 1] - self.mbar[:, None]).sum(axis=1)
        return self.sigma ** 2 * (d + d ** 2 / self.t_est + dm ** 2 / self.ssx)

    def scar(self, t1: int, t2: int) -> NDArray:
        return self.car(t1, t2) / np.sqrt(self.car_variance(t1, t2))


def fit_market_model(est_r: NDArray, est_m: NDArray, win_r: NDArray, win_m: NDArray,
                     window: tuple[int, int] = WINDOW) -> MarketModel:
    """OLS market model per row; inputs (M, T) estimation and (M, W) event-window returns."""
    est_r, est_m = np.atleast_2d(est_r), np.atleast_2d(est_m)
    win_r, win_m = np.atleast_2d(win_r), np.atleast_2d(win_m)
    t = est_r.shape[1]
    mbar = est_m.mean(axis=1)
    rbar = est_r.mean(axis=1)
    xm = est_m - mbar[:, None]
    ssx = (xm ** 2).sum(axis=1)
    beta = (xm * (est_r - rbar[:, None])).sum(axis=1) / ssx
    alpha = rbar - beta * mbar
    resid = est_r - alpha[:, None] - beta[:, None] * est_m
    sigma = np.sqrt((resid ** 2).sum(axis=1) / (t - 2))
    ar = win_r - alpha[:, None] - beta[:, None] * win_m
    return MarketModel(alpha, beta, sigma, t, mbar, ssx, resid, ar, win_m, window)


def event_rows(dates: pd.DatetimeIndex, event_times: pd.Series | list,
               close_utc_hour: float = 20.0) -> NDArray:
    """Row of day 0 in `dates` for each event time (UTC).

    Day 0 is the first trading date d with close(d) >= event time, the close taken as
    `close_utc_hour` UTC on d (16:00 US Eastern daylight time = 20:00 UTC; the hurricane season
    is almost entirely in daylight-saving time). Returns -1 when no such date exists.
    """
    closes = pd.DatetimeIndex(dates).normalize() + pd.Timedelta(hours=close_utc_hour)
    t = pd.DatetimeIndex(pd.to_datetime(pd.Series(event_times)))
    pos = closes.searchsorted(t, side="left")
    return np.where(pos < len(closes), pos, -1)


def extract(returns: NDArray, market: NDArray, day0: NDArray, window: tuple[int, int] = WINDOW,
            est_len: int = EST_LEN, gap: int = GAP) -> tuple[NDArray, ...]:
    """Slice estimation and event windows for asset-events.

    returns: (n_days,) or (n_days, M) aligned with `market` (n_days,); day0: (M,) row of day 0
    for each column (or for the single series). Returns est_r, est_m, win_r, win_m with shapes
    (M, est_len) and (M, W). Raises if a window falls outside the data.
    """
    r = returns if returns.ndim == 2 else returns[:, None]
    day0 = np.atleast_1d(day0)
    if r.shape[1] == 1 and len(day0) > 1:
        r = np.repeat(r, len(day0), axis=1)
    est_start = day0 - gap - est_len
    est_idx = est_start[:, None] + np.arange(est_len)[None, :]
    win_idx = day0[:, None] + np.arange(window[0], window[1] + 1)[None, :]
    if est_idx.min() < 0 or win_idx.max() >= len(market):
        raise ValueError("estimation or event window outside the data")
    cols = np.arange(len(day0))[:, None]
    return r[est_idx, cols], market[est_idx], r[win_idx, cols], market[win_idx]
