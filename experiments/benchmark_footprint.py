#!/usr/bin/env python3
"""Speed of the footprint Monte Carlo: NumPy, the reference C++ kernel and the optimised kernel.

Synthetic inputs: 500 sampled storms (7 nodes over 120 h, 0.5 h steps, so 241 steps), 50 exposure
points on the US Gulf coast, the North Atlantic land mask coarsened to 0.05 degree. After one
warm-up run of each engine, the three engines run interleaved (NumPy, reference, optimised) for
`--reps` repetitions, so that a change in machine load affects all three alike. Reported: median,
minimum and maximum wall time per engine and three ratios of medians, with the range of the ratio
(slowest over fastest run and fastest over slowest). Everything is single-threaded. The 1-minute
load average before and after is recorded: the machine may be shared, so the timings are
indicative. Writes results/benchmark/ (results/benchmark/first_run/ holds the first measurement,
made with the first version of this script at commit fde66f1).

Usage: python experiments/benchmark_footprint.py [--reps 15] [--seed 20261105]
           [--wait-load 1.3 --wait-minutes 10]
With --wait-load, the timing starts once the 1-minute load average is below that value or after
--wait-minutes, whichever comes first (scripts/reproduce.sh uses it; the wait is recorded).
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

ENGINES = {
    "numpy": fp.max_wind_numpy,
    "cpp reference (first kernel)": lambda *a: fp.max_wind_cpp(*a, kernel="reference"),
    "cpp optimised": lambda *a: fp.max_wind_cpp(*a, kernel="optimised"),
}
RATIOS = [("numpy / cpp optimised", "numpy", "cpp optimised"),
          ("cpp reference / cpp optimised", "cpp reference (first kernel)", "cpp optimised"),
          ("numpy / cpp reference", "numpy", "cpp reference (first kernel)")]


def load1() -> float:
    return os.getloadavg()[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--storms", type=int, default=500)
    ap.add_argument("--wait-load", type=float, default=None)
    ap.add_argument("--wait-minutes", type=float, default=10.0)
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
    inputs = (storms, lat_p, lon_p, grid)

    results = {name: f(*inputs) for name, f in ENGINES.items()}  # warm-up and accuracy
    ref = results["numpy"]
    max_rel = {name: float(np.max(np.abs(r - ref) / np.maximum(np.abs(ref), 1e-12)))
               for name, r in results.items() if name != "numpy"}

    waited = time.monotonic()
    if args.wait_load is not None:
        while load1() >= args.wait_load and time.monotonic() - waited < 60 * args.wait_minutes:
            time.sleep(15)
    waited = time.monotonic() - waited
    load_before = load1()
    times: dict[str, list[float]] = {name: [] for name in ENGINES}
    for _ in range(args.reps):
        for name, f in ENGINES.items():
            t0 = time.perf_counter()
            f(*inputs)
            times[name].append(time.perf_counter() - t0)
    load_after = load1()

    steps = int(round((node[-1] - node[0]) / 0.5)) + 1
    evals = n * steps * lat_p.size
    rows = []
    for name, t in times.items():
        rows.append({"engine": name, "reps": args.reps, "median_s": f"{np.median(t):.3f}",
                     "min_s": f"{np.min(t):.3f}", "max_s": f"{np.max(t):.3f}",
                     "ns_per_evaluation": f"{np.median(t) / evals * 1e9:.0f}"})
    for label, a, b in RATIOS:
        ta, tb = times[a], times[b]
        rows.append({"engine": f"ratio {label}", "reps": args.reps,
                     "median_s": f"{np.median(ta) / np.median(tb):.1f}",
                     "min_s": f"{np.min(ta) / np.max(tb):.1f}",
                     "max_s": f"{np.max(ta) / np.min(tb):.1f}", "ns_per_evaluation": ""})
    out = results_dir("benchmark")
    df = pd.DataFrame(rows)
    df.to_csv(out / "footprint_speed.csv", index=False)
    write_manifest(out, args.seed, {
        "inputs": f"synthetic: {n} storms x {steps} steps x {lat_p.size} points = {evals} "
                  "wind evaluations per run",
        "order": "interleaved per repetition: numpy, cpp reference, cpp optimised",
        "threads": "1 (all engines; OMP/OpenBLAS threads set to 1 by the caller)",
        "load1_before": f"{load_before:.2f}", "load1_after": f"{load_after:.2f}",
        "waited_for_load_s": f"{waited:.0f}" + (f" (target below {args.wait_load})"
                                                 if args.wait_load is not None else ""),
        "max_relative_difference_vs_numpy": "; ".join(f"{k}: {v:.2e}"
                                                      for k, v in max_rel.items()),
        "ratio_columns": "median_s = ratio of medians; min_s = min(a)/max(b); "
                         "max_s = max(a)/min(b)",
        "cpu_count": os.cpu_count(), "process_niceness": os.nice(0)})
    print(df.to_string())
    print(f"load {load_before:.2f} -> {load_after:.2f}, max rel diff {max_rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
