#!/usr/bin/env python3
"""Instruction profile of the two C++ footprint kernels with callgrind (valgrind).

Runs build/cpp/profile_driver (built by CMake, Release flags) once per kernel under
`valgrind --tool=callgrind` on 20 synthetic storms x 241 steps x 50 points, and reports the
instructions per wind evaluation and the share of the kernel's instructions spent in each group
of functions (self cost: trigonometry, hypot, pow, exp, asin, sqrt, the kernel's own code).
Instruction counts are deterministic for a given binary and C library, unlike timings; they are
not times (a libm call's instructions are not all equally expensive). Writes
results/benchmark/callgrind.csv. Needs valgrind (callgrind_annotate).

Usage: python experiments/profile_kernel.py [--build build]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

from landfall.provenance import REPO, results_dir, write_manifest

N_STORMS = 20
GROUPS = [("sin and cos", ("s_sin.c", "s_sincos.c", "sincos", "__sin", "__cos")),
          ("hypot", ("hypot",)), ("pow", ("pow",)), ("exp", ("exp",)), ("asin", ("asin",)),
          ("kernel code (incl. inlined)", ("landfall::", "footprint.cpp", "stl_"))]
LINE = re.compile(r"^\s*([\d,]+)\s+\(\s*[\d.]+%\)\s+(\S.*)$")


def group_of(name: str) -> str:
    for g, keys in GROUPS:
        if any(k in name for k in keys):
            return g
    return "other"


def profile(driver: pathlib.Path, kernel: str, tmp: pathlib.Path) -> tuple[dict, int, int]:
    out = tmp / f"cg.{kernel}"
    run = subprocess.run(["valgrind", "--tool=callgrind", f"--callgrind-out-file={out}",
                          str(driver), str(N_STORMS), kernel], capture_output=True, text=True,
                         check=True)
    evals = int(re.search(r"evaluations=(\d+)", run.stdout).group(1))
    ann = subprocess.run(["callgrind_annotate", "--inclusive=no", "--threshold=100", str(out)],
                         capture_output=True, text=True, check=True).stdout
    incl = subprocess.run(["callgrind_annotate", "--inclusive=yes", "--threshold=100", str(out)],
                          capture_output=True, text=True, check=True).stdout
    kernel_fn = "max_wind_reference" if kernel == "reference" else "landfall::max_wind("
    total = max(int(m.group(1).replace(",", "")) for ln in incl.splitlines()
                if (m := LINE.match(ln)) and kernel_fn in m.group(2))
    groups: dict[str, int] = {}
    for ln in ann.splitlines():
        m = LINE.match(ln)
        if not m or "PROGRAM TOTALS" in ln:
            continue
        g = group_of(m.group(2))
        groups[g] = groups.get(g, 0) + int(m.group(1).replace(",", ""))
    return groups, total, evals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=str(REPO / "build"))
    args = ap.parse_args()
    driver = pathlib.Path(args.build) / "cpp" / "profile_driver"
    if not shutil.which("valgrind") or not driver.exists():
        print(f"needs valgrind and {driver} (cmake --build build)")
        return 1
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for kernel in ("reference", "optimised"):
            groups, total, evals = profile(driver, kernel, pathlib.Path(tmp))
            row = {"kernel": kernel, "evaluations": evals,
                   "instructions_per_evaluation": round(total / evals)}
            for g, _ in [*GROUPS, ("other", ())]:
                row[f"share_{g}"] = round(groups.get(g, 0) / total, 3)
            rows.append(row)
    df = pd.DataFrame(rows)
    out = results_dir("benchmark")
    df.to_csv(out / "callgrind.csv", index=False)
    vg = subprocess.run(["valgrind", "--version"], capture_output=True, text=True).stdout.strip()
    write_manifest(out / "callgrind", None, {
        "tool": vg, "driver": "cpp/bench/profile_driver.cpp, built by CMake (Release)",
        "inputs": f"synthetic: {N_STORMS} storms x 241 steps x 50 points",
        "note": "shares are self instructions of each group over the kernel's inclusive count"})
    print(df.T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
