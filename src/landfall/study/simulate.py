"""Simulated return panels with a known abnormal return, for validating the event-study tests.

Each replication has `n_events` event dates and `n_assets` stocks per event (M = n_events *
n_assets asset-events). For every event the simulation draws est_len + gap + 16 trading days:

    R_mt = sigma_m z_mt,   R_it = beta_i R_mt + e_it (+ delta / 11 on days 0..+10),

with beta_i ~ U(0.5, 1.5). Scenarios (`SCENARIOS`):

- gaussian: z and e i.i.d. normal, constant volatility (sigma_m = 1%, sigma_e = 2% daily).
- garch_t: GARCH(1,1) volatility for the market and for each e_it,
  h_t = omega + a x_{t-1}^2 + b h_{t-1} with a = 0.08, b = 0.90 and omega set so that the
  unconditional sd is 1% / 2%, started at the unconditional variance with 100 burn-in days;
  Student-t(5) innovations scaled to unit variance.
- garch_t_event_variance: as garch_t, and e_it is multiplied by 2 on days 0..+10
  (event-induced variance: the variance quadruples in the event window).
- garch_t_cross_correlation: as garch_t, and the stocks of one event share an industry shock
  that the market model does not remove: e_it = sqrt(rho) f_t + sqrt(1 - rho) u_it with rho = 0.3,
  f and u independent GARCH-t series.

The injected abnormal return adds exactly delta to CAR[0, +10] of every asset-event. The residual
sd of the stocks (sigma_e, 2% a day above) can be set per call (`sigma_e`): experiments/
per_stock_power.py uses each real stock's own residual sd.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from landfall.study import bootstrap as bs
from landfall.study import eventstudy as es
from landfall.study import inference as inf


@dataclass(frozen=True)
class Scenario:
    garch: bool
    t_df: float | None
    event_sd_mult: float
    xcorr: float


SCENARIOS = {
    "gaussian": Scenario(False, None, 1.0, 0.0),
    "garch_t": Scenario(True, 5.0, 1.0, 0.0),
    "garch_t_event_variance": Scenario(True, 5.0, 2.0, 0.0),
    "garch_t_cross_correlation": Scenario(True, 5.0, 1.0, 0.3),
}
SIGMA_M, SIGMA_E = 0.01, 0.02
GARCH_A, GARCH_B, BURN = 0.08, 0.90, 100


def _innovations(shape: tuple[int, ...], t_df: float | None, rng: np.random.Generator
                 ) -> NDArray:
    if t_df is None:
        return rng.standard_normal(shape)
    return rng.standard_t(t_df, shape) * np.sqrt((t_df - 2) / t_df)


def noise(shape: tuple[int, ...], sigma: float, sc: Scenario, rng: np.random.Generator
          ) -> NDArray:
    """Series along the last axis: i.i.d. or GARCH(1,1) with unconditional sd `sigma`."""
    if not sc.garch:
        return sigma * _innovations(shape, sc.t_df, rng)
    n = shape[-1]
    z = _innovations((*shape[:-1], n + BURN), sc.t_df, rng)
    omega = sigma ** 2 * (1 - GARCH_A - GARCH_B)
    h = np.full(shape[:-1], sigma ** 2)
    x = np.empty((*shape[:-1], n + BURN))
    for t in range(n + BURN):
        x[..., t] = np.sqrt(h) * z[..., t]
        h = omega + GARCH_A * x[..., t] ** 2 + GARCH_B * h
    return x[..., BURN:]


def simulate_panel(sc: Scenario, n_events: int, n_assets: int, delta: float,
                   rng: np.random.Generator, n_reps: int = 1, est_len: int = es.EST_LEN,
                   gap: int = es.GAP, window: tuple[int, int] = es.WINDOW,
                   sigma_e: float = SIGMA_E) -> tuple[NDArray, ...]:
    """Simulate `n_reps` independent panels at once.

    Returns est_r, est_m (n_reps, M, est_len), win_r, win_m (n_reps, M, W) with asset-events in
    event-major order, and the event group of each asset-event (M,).
    """
    w = window[1] - window[0] + 1
    length = est_len + gap + w
    day0 = est_len + gap - window[0]
    m = noise((n_reps, n_events, length), SIGMA_M, sc, rng)
    e = noise((n_reps, n_events, n_assets, length), sigma_e, sc, rng)
    if sc.xcorr > 0:
        f = noise((n_reps, n_events, 1, length), sigma_e, sc, rng)
        e = np.sqrt(sc.xcorr) * f + np.sqrt(1 - sc.xcorr) * e
    post = slice(day0, day0 + 11)
    e[..., post] *= sc.event_sd_mult
    beta = rng.uniform(0.5, 1.5, (n_reps, n_events, n_assets, 1))
    r = beta * m[:, :, None, :] + e
    r[..., post] += delta / 11.0
    r = r.reshape(n_reps, n_events * n_assets, length)
    mm = np.repeat(m, n_assets, axis=1)
    est = slice(0, est_len)
    win = slice(est_len + gap, length)
    groups = np.repeat(np.arange(n_events), n_assets)
    return r[..., est], mm[..., est], r[..., win], mm[..., win], groups


TESTS = ("naive_t", "patell", "bmp", "bmp_kp", "rank")


def replicate(sc: Scenario, n_events: int, n_assets: int, delta: float,
              rng: np.random.Generator, n_reps: int = 1, n_boot: int = 499,
              window: tuple[int, int] = (0, 10), sigma_e: float = SIGMA_E) -> list[dict]:
    """`n_reps` replications: estimate, p-values of every test and interval coverage of delta.

    The bootstrap is the stationary bootstrap over asset-events in event-major order with mean
    block length n_assets (the stocks of one event tend to fall in one block). With
    `n_boot = 0` no bootstrap is drawn and `boot_cover` is NaN (power experiments).
    """
    est_r, est_m, win_r, win_m, groups = simulate_panel(sc, n_events, n_assets, delta, rng,
                                                        n_reps, sigma_e=sigma_e)
    big = es.fit_market_model(*(a.reshape(-1, a.shape[-1]) for a in (est_r, est_m, win_r,
                                                                       win_m)))
    n_ae = est_r.shape[1]
    car_all, scar_all = big.car(*window), big.scar(*window)
    cols = slice(big.idx(window[0]), big.idx(window[1]) + 1)
    out = []
    for k in range(n_reps):
        rows = slice(k * n_ae, (k + 1) * n_ae)
        car, scar = car_all[rows], scar_all[rows]
        resid = big.resid_est[rows]
        rbar = inf.average_cross_correlation(resid, groups)
        p = {
            "naive_t": inf.naive_t(car).p, "patell": inf.patell(scar, big.t_est).p,
            "bmp": inf.bmp(scar).p, "bmp_kp": inf.bmp_kp(scar, rbar).p,
            "rank": inf.rank_test(resid, big.ar[rows], cols).p,
        }
        lo, hi = inf.t_interval(car)
        boot_cover = float("nan")
        if n_boot > 0:
            idx = bs.stationary_indices(car.size, float(n_assets), n_boot, rng)
            blo, bhi = np.quantile(car[idx].mean(axis=1), [0.025, 0.975])
            boot_cover = blo <= delta <= bhi
        out.append({"estimate": float(car.mean()), "rbar": rbar,
                    **{f"p_{t}": v for t, v in p.items()},
                    "t_cover": lo <= delta <= hi, "boot_cover": boot_cover})
    return out
