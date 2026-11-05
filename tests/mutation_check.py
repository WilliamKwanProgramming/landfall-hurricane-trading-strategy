#!/usr/bin/env python3
"""Mutation check: does the test suite notice deliberate bugs?

Copies src/, tests/, cpp/, scripts/, docs/, results/, README.md, CMakeLists.txt and
pyproject.toml into a temporary directory (the tests read committed results), builds
the C++ module there once, and then applies ONE deliberate bug ("mutant") at a time to the copy,
rebuilds the module if a C++ file changed, and runs the whole pytest suite on the copy. A mutant
is "caught" if at least one test fails. The working tree is never modified. Mutants are chosen
where a bug would silently produce plausible numbers. Exits with status 1 if a mutant survives or
the unmutated baseline fails. Downloaded data are read from <repo>/data (LANDFALL_DATA).

Usage: python tests/mutation_check.py [--out results/mutation_check.txt]
Needs cmake, ninja, a C++17 compiler, pybind11 >= 3.0 and the test dependencies.
"""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
COPY = ["src", "tests", "cpp", "scripts", "docs", "results", "README.md", "CMakeLists.txt",
        "pyproject.toml"]
ES = "src/landfall/study/eventstudy.py"
INF = "src/landfall/study/inference.py"
FP = "src/landfall/footprint.py"
PR = "src/landfall/prices.py"
PL = "src/landfall/pipeline.py"
BT = "src/landfall/study/backtest.py"
SIM = "src/landfall/study/simulate.py"
PW = "src/landfall/study/power.py"

# (name, [(file, text that must occur exactly once, replacement), ...])
MUTANTS = [
    ("M1 estimation window overlaps the event window (no 30-day gap)", [
        (ES, "est_start = day0 - gap - est_len", "est_start = day0 - est_len")]),
    ("M2 off-by-one in the event-day index", [
        (ES, "        return day - self.window[0]\n",
         "        return day - self.window[0] + 1\n")]),
    ("M3 abnormal return with the wrong sign", [
        (ES, "ar = win_r - alpha[:, None] - beta[:, None] * win_m",
         "ar = alpha[:, None] + beta[:, None] * win_m - win_r")]),
    ("M4 forecast features use the position 12/24 h after t instead of before", [
        ("src/landfall/forecast.py",
         "e, n = displacement_km(at(lat, -h), at(lon, -h), lat0, lon0)",
         "e, n = displacement_km(lat0, lon0, at(lat, h), at(lon, h))")]),
    ("M5 training seasons include the test season", [
        ("src/landfall/forecast.py",
         '(samples["season"] < test_season)', '(samples["season"] <= test_season)')]),
    ("M6 residual variance with T - 1 instead of T - 2 degrees of freedom", [
        (ES, "sigma = np.sqrt((resid ** 2).sum(axis=1) / (t - 2))",
         "sigma = np.sqrt((resid ** 2).sum(axis=1) / (t - 1))")]),
    ("M7 bootstrap block length ignored (always i.i.d.)", [
        ("src/landfall/study/bootstrap.py", "p_new = 1.0 / mean_block", "p_new = 1.0")]),
    ("M8 trading cost charged once instead of on entry and exit", [
        ("src/landfall/study/backtest.py",
         "trading = 2.0 * (costs.stock_bps", "trading = 1.0 * (costs.stock_bps")]),
    ("M9 Sharpe annualised with k instead of sqrt(k)", [
        ("src/landfall/study/sharpe.py", "return float(sr * np.sqrt(periods_per_year))",
         "return float(sr * periods_per_year)")]),
    ("M10 deflated Sharpe ratio ignores the number of trials", [
        ("src/landfall/study/sharpe.py",
         "return psr(sr, t, skew, kurt, expected_max_sharpe(n_trials, var_sr))",
         "return psr(sr, t, skew, kurt, 0.0)")]),
    ("M11 inland decay starts 6 h before landfall (NumPy)", [
        (FP, "after = steps >= first[:, None]", "after = steps >= first[:, None] - 12")]),
    ("M12 Holland profile uses exponent 1/B instead of B", [
        (FP, "x = (np.asarray(rm_km, dtype=float) / r) ** np.asarray(b, dtype=float)",
         "x = (np.asarray(rm_km, dtype=float) / r) ** (1.0 / np.asarray(b, dtype=float))")]),
    ("M13 Kolari-Pynnonen factor without the (N - 1) term", [
        (INF, "return float(np.sqrt((1 - rbar) / (1 + (n - 1) * rbar)))",
         "return float(np.sqrt((1 - rbar) / (1 + rbar)))")]),
    ("M14 rank test centred at L/2 instead of (L + 1)/2", [
        (INF, "u = (ranks - (length + 1) / 2.0).mean(axis=0)",
         "u = (ranks - length / 2.0).mean(axis=0)")]),
    ("M15 Holm replaced by Bonferroni", [
        ("src/landfall/study/multiple.py", "adj = np.minimum(1.0, (m - np.arange(m)) * p[order])",
         "adj = np.minimum(1.0, m * p[order])")]),
    ("M16 forecast public at its synoptic time (availability delay ignored)", [
        ("src/landfall/data/atcf.py",
         'avail = rows["issue_time"] + pd.Timedelta(hours=delay_hours)',
         'avail = rows["issue_time"]')]),
    ("M17 placebo dates not kept away from landfalls", [
        ("src/landfall/realstudy.py", "                if near:\n                    continue",
         "                if False:\n                    continue")]),
    ("M18 HURDAT2 west longitudes read as east", [
        ("src/landfall/data/hurdat2.py",
         '    if token[-1] == "W":\n        return -v\n    if token[-1] != "E":',
         '    if token[-1] == "W":\n        return v\n    if token[-1] != "E":')]),
    ("M19 translation asymmetry factor k(Rm) = 1 instead of 1/2 (NumPy)", [
        (FP, "k = np.where(d > 0, rm * d / (rm ** 2 + d ** 2), 0.0)",
         "k = np.where(d > 0, 2.0 * rm * d / (rm ** 2 + d ** 2), 0.0)")]),
    ("C1 C++ kernel: inland decay rate 0.95 / h instead of 0.095 / h", [
        ("cpp/src/footprint.cpp", "constexpr double kKdAlpha = 0.095;",
         "constexpr double kKdAlpha = 0.95;")]),
    ("C2 C++ reference kernel: wind direction clockwise (southern-hemisphere rotation)", [
        ("cpp/src/footprint.cpp", "\n  const double ue = -vs * cos_th + k * ce;",
         "\n  const double ue = vs * cos_th + k * ce;"),
        ("cpp/src/footprint.cpp", "\n  const double un = vs * sin_th + k * cn;",
         "\n  const double un = -vs * sin_th + k * cn;")]),
    ("P1 own-prices loader accepts a zero price", [
        (PR, "bad = num.notna() & ~(np.isfinite(num) & (num > 0))",
         "bad = num.notna() & ~(np.isfinite(num) & (num >= 0))")]),
    ("P2 own-prices loader accepts dates out of order", [
        (PR, "    if back.size:\n", "    if False:\n")]),
    ("P3 own-prices alignment forward-fills missing prices instead of using common dates", [
        (PR, "both = prices[[asset, market]].dropna()",
         "both = prices[[asset, market]].ffill().dropna()")]),
    ("P4 sign table reads '-' as a positive hypothesis", [
        (PL, 'SIGNS = {"+": 1, "-": -1,', 'SIGNS = {"+": 1, "-": 1,')]),
    ("P5 placebo dates mapped to the next asset date after the drawn one (off by one)", [
        (PL, 'calendar.to_numpy()[p0], side="left")', 'calendar.to_numpy()[p0], side="right")')]),
    ("P6 two-sided assets pooled and traded as if signed", [
        (PL, 'return self.group == "storm" and self.sign != 0', 'return self.group == "storm"')]),
    ("C3 optimised C++ kernel: sign error in the half-angle sine of the longitude difference", [
        ("cpp/src/footprint.cpp", "const double s_dl = slp[p] * clc - clp[p] * slc;",
         "const double s_dl = slp[p] * clc + clp[p] * slc;")]),
    ("C4 optimised C++ kernel: sine and cosine of the bearing swapped", [
        ("cpp/src/footprint.cpp",
         "const double cos_th = h > 0.0 ? x / h : 1.0;\n"
         "        const double sin_th = h > 0.0 ? y / h : 0.0;",
         "const double cos_th = h > 0.0 ? y / h : 1.0;\n"
         "        const double sin_th = h > 0.0 ? x / h : 0.0;")]),
    ("C5 optimised C++ kernel: cos(dl) = 1 - s^2 instead of 1 - 2 s^2 (half-angle identity)", [
        ("cpp/src/footprint.cpp", "(1.0 - 2.0 * s_dl * s_dl)", "(1.0 - s_dl * s_dl)")]),
    ("C6 optimised C++ kernel: per-point latitude sine taken from the half angle", [
        ("cpp/src/footprint.cpp", "    sp[p] = std::sin(phi);",
         "    sp[p] = std::sin(0.5 * phi);")]),
    ("R1 forward-only selection also looks at the season it trades", [
        (BT, "earlier = results[n][results[n][season_col] < s][value].dropna()",
         "earlier = results[n][results[n][season_col] <= s][value].dropna()")]),
    ("R2 gross-return statistics computed from the net return", [
        (PL, 'g = evt["gross"].to_numpy()', 'g = evt["net"].to_numpy()')]),
    ("R3 split check with zero tolerance (an adjusted split ratio is never matched)", [
        (PR, "SPLIT_TOL = 0.03", "SPLIT_TOL = 0.0")]),
    ("R4 simulator ignores the requested residual volatility", [
        (SIM, "e = noise((n_reps, n_events, n_assets, length), sigma_e, sc, rng)",
         "e = noise((n_reps, n_events, n_assets, length), SIGMA_E, sc, rng)")]),
    ("R5 power curve does not pass its residual volatility to the simulator", [
        (PW, "n_boot=0, sigma_e=sigma)", "n_boot=0)")]),
    ("R6 plug-in t-test power uses n instead of n - 1 degrees of freedom for the critical value", [
        (PW, "tc = stats.t.ppf(0.975, n - 1)", "tc = stats.t.ppf(0.975, n)")]),
]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds per pytest run")
    args = ap.parse_args()
    lines: list[str] = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    commit = run(["git", "-C", str(REPO), "rev-parse", "HEAD"]).stdout.strip() or "unknown"
    dirty = run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "tests", "cpp",
                 "scripts", "CMakeLists.txt", "pyproject.toml"]).stdout
    import numpy

    log("# Generated by tests/mutation_check.py; do not edit by hand.")
    log("# command: python tests/mutation_check.py" + (f" --out {args.out}" if args.out else ""))
    log(f"# git:     {commit} ({'uncommitted changes' if dirty else 'clean'})")
    log(f"# date:    {datetime.datetime.now(datetime.UTC):%Y-%m-%dT%H:%M:%SZ}")
    log(f"# python:  {platform.python_version()}; numpy {numpy.__version__}")
    log("")

    survived = 0
    with tempfile.TemporaryDirectory(prefix="landfall_mutation_") as tmp:
        work = pathlib.Path(tmp) / "src"
        build = pathlib.Path(tmp) / "build"
        for item in COPY:
            src = REPO / item
            if src.is_dir():
                shutil.copytree(src, work / item, ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", ".pytest_cache", "*.egg-info"))
            else:
                shutil.copy2(src, work / item)
        cfg = run(["cmake", "-S", str(work), "-B", str(build), "-G", "Ninja",
                   f"-DPython3_EXECUTABLE={sys.executable}"])
        if cfg.returncode != 0 or "landfall_cpp: skipped" in cfg.stdout:
            log("configure failed or landfall_cpp skipped:\n" + cfg.stdout[-2000:])
            return 1

        def build_cpp():
            return run(["cmake", "--build", str(build), "--target", "landfall_cpp"])

        env = dict(os.environ, PYTHONPATH=str(work / "src"), LANDFALL_DATA=str(REPO / "data"),
                   LANDFALL_CPP_DIR=str(build / "cpp"), LANDFALL_REQUIRE_CPP="1")
        where = run([sys.executable, "-c", "import landfall; print(landfall.__file__)"],
                    env=env, cwd=work).stdout.strip()
        if not where.startswith(str(work)):
            log(f"the copy is not the package under test ({where}); aborting")
            return 1

        def pytest_run():
            t0 = time.monotonic()
            try:
                p = run([sys.executable, "-m", "pytest", "tests", "-q", "-x", "--tb=no",
                         "-p", "no:cacheprovider"], env=env, cwd=work, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                return None, ["(timeout)"], time.monotonic() - t0, "timeout"
            failed = [ln.split(" - ")[0].replace("FAILED ", "")
                      for ln in p.stdout.splitlines() if ln.startswith("FAILED")]
            summary = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-300:]
            return p.returncode, failed, time.monotonic() - t0, summary

        if build_cpp().returncode != 0:
            log("baseline C++ build failed")
            return 1
        rc, _, secs, summary = pytest_run()
        log(f"baseline (no mutant): {summary} [{secs:.0f} s]")
        if rc != 0:
            log("baseline must pass; aborting")
            return 1
        log("")

        originals: dict[str, str] = {}
        for name, edits in MUTANTS:
            touched: dict[str, str] = {}
            try:
                for rel, old, new in edits:
                    if rel not in originals:
                        originals[rel] = (work / rel).read_text()
                    text = touched.get(rel, originals[rel])
                    if text.count(old) != 1:
                        log(f"{name}: SETUP ERROR, pattern found {text.count(old)} times in {rel}")
                        return 1
                    touched[rel] = text.replace(old, new)
                for rel, text in touched.items():
                    (work / rel).write_text(text)
                if any(r.startswith("cpp/") for r in touched) and build_cpp().returncode != 0:
                    log(f"{name}: caught at build time (does not compile)")
                    continue
                rc, failed, secs, summary = pytest_run()
                caught = rc != 0
                survived += not caught
                log(f"{name}: {'caught' if caught else 'SURVIVED'} - {summary} [{secs:.0f} s]")
                for f in failed[:2]:
                    log(f"    first failing test: {f}")
            finally:
                for rel in touched:
                    (work / rel).write_text(originals[rel])
                if any(r.startswith("cpp/") for r in touched):
                    build_cpp()

    log("")
    log(f"{len(MUTANTS) - survived} of {len(MUTANTS)} mutants caught.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n")
        print(f"wrote {args.out}")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
