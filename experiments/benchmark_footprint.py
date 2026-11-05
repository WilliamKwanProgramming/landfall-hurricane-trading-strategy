#!/usr/bin/env python3
"""Speed of the footprint Monte Carlo: C++ kernel vs NumPy reference, on identical inputs.

Synthetic inputs: 500 sampled storms (7 nodes over 120 h, 0.5 h steps), 50 exposure points
on the US Gulf coast, the North Atlantic land mask coarsened to 0.05 degree. Each engine is timed
`--reps` times (after one warm-up run); reported: median, minimum and maximum wall time and the
ratio of medians. The 1-minute load average before and after is recorded, because timings are
only meaningful on a quiet machine. Writes results/benchmark/.

Usage: python experiments/benchmark_footprint.py [--reps 15] [--seed 20261105]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from landfall import footprint as fp
from landfall.geo import atlantic_land_grid
from landfall.provenance import results_dir, write_manifest


def load1() -> float:
    return os.getloadavg()[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--storms", type=int, default=500)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    n = args.storms
    node = np.array([0.0, 12, 24, 48, 72, 96, 120])
    lat = 24 + np.cumsum(rng.normal(0.5, 0.4, (n, 7)), axis=1)
    lon = -88 + np.cumsum(rng.normal(0.3, 0.6, (n, 7)), axis=1)
    vmax = rng.uniform(60, 140, (n, 7))
    storms = fp.StormSet(node, lat, lon, vmax, rng.uniform(20, 60, n), rng.uniform(1.0, 2.2, n))
    lat_p = rng.uniform(26, 31, 50)
    lon_p = rng.uniform(-97, -82, 50)
    grid = atlantic_land_grid(coarsen=6)

    a = fp.max_wind_numpy(storms, lat_p, lon_p, grid)
    b = fp.max_wind_cpp(storms, lat_p, lon_p, grid)
    max_rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-12)))

    load_before = load1()
    times = {"numpy": [], "cpp": []}
    for _ in range(args.reps):
        for name, f in (("numpy", fp.max_wind_numpy), ("cpp", fp.max_wind_cpp)):
            t0 = time.perf_counter()
            f(storms, lat_p, lon_p, grid)
            times[name].append(time.perf_counter() - t0)
    load_after = load1()
    rows = []
    for name, t in times.items():
        rows.append({"engine": name, "reps": args.reps, "median_s": f"{np.median(t):.4f}",
                     "min_s": f"{np.min(t):.4f}", "max_s": f"{np.max(t):.4f}"})
    ratio = np.median(times["numpy"]) / np.median(times["cpp"])
    lo = np.min(times["numpy"]) / np.max(times["cpp"])
    hi = np.max(times["numpy"]) / np.min(times["cpp"])
    rows.append({"engine": "ratio numpy/cpp", "reps": args.reps, "median_s": f"{ratio:.1f}",
                 "min_s": f"{lo:.1f}", "max_s": f"{hi:.1f}"})
    out = results_dir("benchmark")
    df = pd.DataFrame(rows)
    df.to_csv(out / "footprint_speed.csv", index=False)
    write_manifest(out, args.seed, {
        "inputs": f"synthetic: {n} storms x 241 steps x 50 points", "threads": "1 (both)",
        "load1_before": f"{load_before:.2f}", "load1_after": f"{load_after:.2f}",
        "max_relative_difference_cpp_vs_numpy": f"{max_rel:.2e}",
        "cpu_count": os.cpu_count()})
    print(df.to_string())
    print(f"load {load_before:.2f} -> {load_after:.2f}, max rel diff {max_rel:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
