#!/usr/bin/env python3
"""Size of the placebo test of the protocols on null synthetic prices (a check of the checker).

For each replication: synthetic prices without any abnormal return (landfall.prices.
synthetic_prices, a new seed per replication) for four assets, the 39 primary events of
results/real_eventstudy/events.csv, the placebo of docs/protocol.md (dates in the same calendar
week of other years, away from landfalls) with `--sets` placebo sets, and the two-sided placebo
p-value of each asset's mean CAR[0, +10]. Reported: the share of p-values below 0.05 and 0.10 with
binomial Monte Carlo standard errors (asset-level p-values within a replication share the
market and are treated as independent, which slightly understates the standard error).
Synthetic data. Writes results/placebo_size/.

Usage: python experiments/placebo_size.py [--reps 60] [--sets 200] [--seed 20261105]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from landfall import pipeline as pl
from landfall import prices as P
from landfall import realstudy as rs
from landfall.provenance import REPO, results_dir, write_manifest

NAMES = ["A", "B", "C", "D"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--sets", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261105)
    args = ap.parse_args()
    events = pd.read_csv(REPO / "results" / "real_eventstudy" / "events.csv",
                         parse_dates=["time"])
    prim = events[events["kept_primary"]].sort_values("time").reset_index(drop=True)
    excl = pd.read_csv(REPO / "results" / "landfall_counts" /
                       "us_landfalls_any_intensity_1985_2024.csv", parse_dates=["time"])
    rows = []
    for rep in range(args.reps):
        prices = P.synthetic_prices(NAMES, "M", prim["time"], seed=args.seed + rep)
        datas = []
        for n in NAMES:
            r = P.asset_returns(prices, n, "M")
            datas.append(pl.AssetData(pl.Asset(n, "control", 0), r.dates, r.ret, r.mkt))
        per_asset, _, _, _ = pl.event_study(datas, prim["time"], args.seed)
        post = per_asset[per_asset.window == "post[0,+10]"].set_index("asset")
        cal = pd.DatetimeIndex(prices["M"].index[1:])
        plc, _ = pl.placebo(datas, cal, prim["time"], excl["time"], (1991, 2022), args.sets,
                            args.seed + rep)
        for n in NAMES:
            le, p = rs.placebo_position(post.loc[n, "mean_car"], plc[n])
            rows.append({"rep": rep, "asset": n, "mean_car": post.loc[n, "mean_car"],
                         "bmp_p": post.loc[n, "bmp_p"], "placebo_mean": np.nanmean(plc[n]),
                         "placebo_sd": np.nanstd(plc[n], ddof=1), "placebo_p": p})
        if (rep + 1) % 10 == 0:
            print(f"{rep + 1} of {args.reps} replications", flush=True)
    d = pd.DataFrame(rows)
    out = results_dir("placebo_size")
    d.to_csv(out / "placebo_pvalues.csv", index=False, float_format="%.6g")
    summ = []
    for test in ("placebo_p", "bmp_p"):
        for a in (0.05, 0.10):
            rate = float((d[test] < a).mean())
            summ.append({"test": test.replace("_p", ""), "level": a, "tests": len(d),
                         "rejection_rate": round(rate, 3),
                         "mc_se": round(float(np.sqrt(rate * (1 - rate) / len(d))), 3)})
    s = pd.DataFrame(summ)
    s.to_csv(out / "summary.csv", index=False)
    write_manifest(out, args.seed, {"data": "SYNTHETIC null prices", "reps": args.reps,
                                    "placebo_sets": args.sets, "assets_per_rep": len(NAMES)})
    print(s.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
