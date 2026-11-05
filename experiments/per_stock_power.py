#!/usr/bin/env python3
"""Experiment (e): power of the per-stock tests at each real stock's own volatility.

Added after an independent review of the first version of this repository. The power quoted there
for one stock (44% at a true 2% CAR, `results/synthetic_eventstudy/garch_t_35x1.csv`) was
simulated with a residual sd of 2% a day for every stock, but the residual sd of seven of the
eight stocks is about 1.0-1.5% a day (only RRC is far above 2%), so it understates their power and
overstates that of RRC.

Two numbers per stock come from the real data of the primary event study: the number of events
that passed the data screen and the root mean square, over those events, of the residual sd of
the market model (days -280..-31), i.e. the stock's average residual variance (the median is
lower; both are stored). With it the simulated CAR[0, +10] has about the standard deviation of
the stock's real CARs. Everything else is simulated (landfall.study.simulate, scenario garch_t:
GARCH(1,1) with Student-t(5) innovations, market sd 1% a day): one stock, that many events,
unconditional residual sd set to that number, a true abnormal return delta added to CAR[0, +10],
the same tests as in experiments/synthetic_eventstudy.py. Reported: rejection rate at 5% of each
test for delta in DELTAS (delta = 0 is the size), and the smallest tested CAR that the BMP test
detects with probability 80% (linear interpolation between grid points). As a cross-check, the
power of a plain two-sided t test at a 2% CAR given each stock's own sample standard error of
the mean real CAR (noncentral t; it treats that standard error as known and includes any
unusual event in the stock's real CARs, so it can be lower than the simulated power).

Usage: python experiments/per_stock_power.py [--reps 2000] [--seed 20261105]
Needs results/real_eventstudy/events.csv (experiments/real_eventstudy.py) and the skfolio prices.
Writes results/per_stock_power/{power.csv, summary.csv, MANIFEST.txt}.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from real_eventstudy import asset_data

from landfall import pipeline as pl
from landfall.data.prices import load_skfolio_prices, simple_returns
from landfall.provenance import results_dir, write_manifest
from landfall.study.power import interpolate_target, plugin_t_power, power_curve

DELTAS = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06)
POWER_TARGET = 0.8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261105)
    args = ap.parse_args()

    prices, index = load_skfolio_prices()
    ret = simple_returns(prices)
    mkt = simple_returns(index).to_numpy()
    events = pd.read_csv("results/real_eventstudy/events.csv", parse_dates=["time"])
    prim = events[events["kept_primary"]].sort_values("time").reset_index(drop=True)
    datas = asset_data(ret, mkt)
    fits, _ = pl.fit_assets(datas, prim["time"])

    long_rows, summary = [], []
    for i, d in enumerate(datas):
        name = d.asset.name
        f = fits[name]
        n, sd = len(f.events), f.mm.sigma
        sigma = float(np.sqrt(np.mean(sd ** 2)))
        car = f.mm.car(0, 10)
        se_real = float(car.std(ddof=1) / np.sqrt(n))
        curve = power_curve(n, sigma, DELTAS, args.reps, [args.seed, 5, i])
        for crow in curve.to_dict("records"):
            long_rows.append({"stock": name, "group": d.asset.group, "events": n,
                              "resid_sd_rms": sigma, **crow})
        power = curve["reject_bmp"].to_numpy()
        pick = {f"power_bmp_{int(round(100 * x))}pct": float(power[DELTAS.index(x)])
                for x in (0.01, 0.02, 0.03)}
        summary.append({
            "stock": name, "group": d.asset.group, "events": n,
            "resid_sd_rms": sigma, "resid_sd_median": float(np.median(sd)),
            "resid_sd_min": float(sd.min()), "resid_sd_max": float(sd.max()),
            "real_car_sd": float(car.std(ddof=1)), "size_bmp": float(power[0]), **pick,
            "delta_80pct_power_bmp": interpolate_target(np.asarray(DELTAS), power, POWER_TARGET),
            "real_car_se": se_real, "plugin_t_power_2pct": plugin_t_power(se_real, n, 0.02)})
        print(f"{name}: n={n} sd(rms)={100 * sigma:.2f}% size={power[0]:.3f} "
              f"power 1/2/3% = {pick['power_bmp_1pct']:.3f}/{pick['power_bmp_2pct']:.3f}/"
              f"{pick['power_bmp_3pct']:.3f}", flush=True)

    out = results_dir("per_stock_power")
    pd.DataFrame(long_rows).to_csv(out / "power.csv", index=False, float_format="%.5f")
    summ = pd.DataFrame(summary)
    summ.to_csv(out / "summary.csv", index=False, float_format="%.5f")
    write_manifest(out, args.seed, {
        "data": "stock-specific event counts and rms residual sd from the REAL primary event "
                "study; returns simulated (landfall.study.simulate, scenario garch_t)",
        "design": f"1 stock x n events, {args.reps} replications per cell, window [0,+10]",
        "deltas": ", ".join(f"{x:.3f}" for x in DELTAS),
        "rng": "numpy default_rng([seed, 5, stock index, delta index])"})
    print(summ.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
