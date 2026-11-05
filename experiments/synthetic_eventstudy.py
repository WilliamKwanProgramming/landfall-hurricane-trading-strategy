#!/usr/bin/env python3
"""Experiment (a): do the event-study statistics recover a known abnormal return?

Synthetic data only (landfall.study.simulate): 30 events x 4 stocks = 120 asset-events per
replication, market model with a 250-day estimation window ending 30 days before day 0, and an
abnormal return delta added to CAR[0, +10]. For each scenario and delta in {0, 0.5%, 1%, 2%}:
bias and RMSE of the mean CAR, rejection rate at 5% of each test (size when delta = 0, power
otherwise), and coverage of the 95% t interval and of the stationary-bootstrap interval
(499 resamples, mean block 4). Monte Carlo standard errors: sqrt(p (1 - p) / reps) for rates,
sd / sqrt(reps) for the bias.

Usage: python experiments/synthetic_eventstudy.py --scenario garch_t [--reps 2000] [--seed N]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from landfall.provenance import results_dir, write_manifest
from landfall.study import simulate as sim

DELTAS = (0.0, 0.005, 0.01, 0.02)
N_EVENTS, N_ASSETS, BATCH = 30, 4, 100


def summarise(df: pd.DataFrame, delta: float) -> dict:
    reps = len(df)
    err = df["estimate"] - delta
    row = {"delta": delta, "reps": reps, "mean_estimate": df["estimate"].mean(),
           "bias": err.mean(), "bias_mcse": err.std(ddof=1) / np.sqrt(reps),
           "rmse": float(np.sqrt((err ** 2).mean())), "mean_rbar": df["rbar"].mean()}
    for t in sim.TESTS:
        p = float((df[f"p_{t}"] < 0.05).mean())
        row[f"reject_{t}"] = p
        row[f"reject_{t}_mcse"] = np.sqrt(p * (1 - p) / reps)
    for c in ("t_cover", "boot_cover"):
        p = float(df[c].mean())
        row[c] = p
        row[f"{c}_mcse"] = np.sqrt(p * (1 - p) / reps)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=sorted(sim.SCENARIOS))
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261105)
    args = ap.parse_args()
    sc = sim.SCENARIOS[args.scenario]
    rows = []
    for k, delta in enumerate(DELTAS):
        rng = np.random.default_rng([args.seed, sorted(sim.SCENARIOS).index(args.scenario), k])
        res = []
        while len(res) < args.reps:
            res += sim.replicate(sc, N_EVENTS, N_ASSETS, delta, rng,
                                 n_reps=min(BATCH, args.reps - len(res)))
        rows.append(summarise(pd.DataFrame(res), delta))
    out = results_dir("synthetic_eventstudy")
    table = pd.DataFrame(rows)
    table.to_csv(out / f"{args.scenario}.csv", index=False, float_format="%.5f")
    write_manifest(out / "manifests" / args.scenario, args.seed, {
        "data": "synthetic (landfall.study.simulate)", "scenario": args.scenario,
        "design": f"{N_EVENTS} events x {N_ASSETS} stocks, window [0,+10]",
        "rng": "numpy default_rng([seed, scenario index, delta index])"})
    print(table.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
