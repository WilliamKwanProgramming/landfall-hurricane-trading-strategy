#!/usr/bin/env python3
"""Check that the numbers in README.md are the numbers in the committed results files.

Five checks:
1. Every generated table between <!-- table:KEY --> markers in README.md is identical to the
   table KEY in results/tables.md (written by scripts/make_tables.py).
2. Each entry of CHECKS names a results file, a text that must occur in it and a text that must
   occur in README.md (outside the generated tables: numbers quoted in prose). When a result is
   regenerated and its number changes, the first part fails; when the README is edited by hand
   and no longer quotes the number, the second fails.
3. Each entry of DERIVED computes the numbers of a sentence from the rows of a results file (a
   minimum, a range, a count) and requires the README to contain the sentence with them filled in.
4. Each entry of FACTS names a claim that must hold in the rows of a results file and a text that
   must occur in the README.
5. Every interval written in the README prose as "x [lo, hi]" occurs in results/tables.md, and
   every relative link or image points to a file that exists.

Usage: python scripts/check_readme_numbers.py      (exit status 1 on any problem)
Standard library only.
"""

from __future__ import annotations

import csv
import math
import pathlib
import re
import sys
from collections.abc import Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
RE = "results/real_eventstudy"
SY = "results/synthetic_eventstudy"
LC = "results/landfall_counts"
BM = "results/benchmark"
PS = "results/per_stock_power"
OP = "results/own_prices_synthetic"

CHECKS = [
    # Real-data event study: the pooled signed test, the forecast-trigger rule, the placebo.
    (f"{RE}/MANIFEST.txt", "events_primary: 39", "39 US hurricane landfalls"),
    (f"{RE}/MANIFEST.txt", "events_total: 48", "all 48 storms kept"),
    (f"{RE}/placebo.csv", '"storm (signed, pooled)",-0.00229995,', "-0.23%"),
    (f"{RE}/placebo.csv", ",0.478,0.956", "placebo p = 0.96"),
    (f"{RE}/trading_loso.csv", "R2 forecast trigger,152,-0.00351724,-0.00629465,-0.000960345",
     "-0.35% [-0.63, -0.10]"),
    (f"{RE}/trading_dsr.csv", ",148,0.0124617", "deflated Sharpe ratio of 0.01"),
    # Power of the pooled design (synthetic GARCH-t simulations, 30 events x 4 stocks).
    (f"{SY}/garch_t.csv", ",0.40500,0.01098,", "about 40%"),
    (f"{SY}/garch_t.csv", ",0.89600,0.00683,", "about 90%"),
    # Landfall definitions: facts about the storms found by one definition only.
    (f"{LC}/storm_level_disagreements.csv", "AL041970,geometric,CELIA,1970,0,", "Celia 1970"),
    (f"{LC}/storm_level_agreement_1851_2019.csv", "geometric_only,16", "12 of the 16"),
    (f"{LC}/storm_level_agreement_1851_2019.csv", "flag_only,22",
     "Of the 22 found only by the flags"),
    (f"{LC}/storm_level_disagreements.csv", "AL052019,flag,DORIAN,2019,7,160.0,85.0,",
     "Dorian 2019, flagged at 85 kt"),
    # Footprint kernel benchmark (indicative timings; they change when the benchmark is rerun).
    (f"{BM}/footprint_speed.csv", "ratio numpy / cpp optimised,15,2.4,1.6,3.4",
     "2.4 times faster than NumPy"),
    (f"{BM}/footprint_speed.csv", "ratio numpy / cpp optimised,15,2.4,1.6,3.4",
     "the fastest NumPy run over the slowest optimised run is 1.6, the slowest over the fastest "
     "3.4"),
    (f"{BM}/footprint_speed.csv", "ratio cpp reference / cpp optimised,15,2.4,",
     "and than the first kernel"),
    (f"{BM}/footprint_speed.csv", "ratio numpy / cpp reference,15,1.0,",
     "The first kernel again ran at about NumPy's speed (ratio 1.0)"),
    (f"{BM}/first_run/footprint_speed.csv", "ratio numpy/cpp,15,1.1,0.7,1.5",
     "ratio of medians 1.1 in the first run, 1.0 in the second"),
    (f"{BM}/MANIFEST.txt", "load1_before: 1.00", "load average was 1.00 before and after"),
    (f"{BM}/MANIFEST.txt", "load1_after: 1.00", "load average was 1.00 before and after"),
    (f"{BM}/MANIFEST.txt", "process_niceness: 19", "at nice 19"),
    (f"{BM}/MANIFEST.txt", "cpp optimised: 2.94e-14", "largest relative difference is 2.94e-14"),
    (f"{LC}/definitions.csv", '"US coast, vmax >= 64 kt, 1851-2019",333,336,', "333 flagged, 336"),
    # The kernel profile (callgrind instruction counts).
    (f"{BM}/callgrind.csv", "reference,241000,1044,0.408,0.153,", "1,044 instructions"),
    (f"{BM}/callgrind.csv", "reference,241000,1044,0.408,0.153,", "41% of them in sine"),
    (f"{BM}/callgrind.csv", "reference,241000,1044,0.408,0.153,", "15% in `hypot`"),
    (f"{BM}/callgrind.csv", "optimised,241000,376,0.022,0.003,0.341,", "376 instructions"),
    (f"{BM}/callgrind.csv", "optimised,241000,376,0.022,0.003,0.341,", "34% of them in `pow`"),
    # Placebo size on null synthetic prices.
    ("results/placebo_size/summary.csv", "placebo,0.05,240,0.158,0.024",
     "15.8% of tests (Monte Carlo SE 2.4)"),
    ("results/placebo_size/summary.csv", "bmp,0.05,240,0.05,0.014", "BMP holds 5.0%"),
]

Rows = list[dict[str, str]]
_cache: dict[str, Rows] = {}


def rows(path: str) -> Rows:
    if path not in _cache:
        with (ROOT / path).open(newline="", encoding="utf-8") as fh:
            _cache[path] = list(csv.DictReader(fh))
    return _cache[path]


def num(r: dict[str, str], col: str) -> float:
    return float(r[col])


def pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}"


def pct_ci(r: dict[str, str], mean: str, lo: str, hi: str) -> str:
    """A mean and its interval in percent, as the README writes them: -0.15% [-0.43, 0.11]."""
    return f"{pct(num(r, mean))}% [{pct(num(r, lo))}, {pct(num(r, hi))}]"


def where(rs: Rows, **eq: str) -> dict[str, str]:
    (hit,) = [r for r in rs if all(r[k] == v for k, v in eq.items())]
    return hit


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
P4 = ("bmp_p", "patell_p", "naive_p", "rank_p")  # the four per-stock tests
SENSITIVITY_FLAGS = ("RRC", "flag events 1990-2019")
SENSITIVITY_ALL = ("MRK", "all events kept")
THREE_FLAG_ONLY = {("BARRY", "2019"), ("CINDY", "2005"), ("GASTON", "2004")}


def r2(rs: Rows) -> dict[str, str]:
    return where(rs, rule="R2 forecast trigger")


def r1(rs: Rows) -> dict[str, str]:
    return where(rs, rule="R1 calendar (not tradeable)")


def stock(rs: Rows, name: str) -> dict[str, str]:
    return where(rs, stock=name)


def min_p(rs: Rows, cols: tuple[str, ...] = P4) -> float:
    return min(num(r, c) for r in rs for c in cols)


def others(rs: Rows, name: str = "RRC") -> Rows:
    """The per-stock power rows of the stocks other than `name`."""
    return [r for r in rs if r["stock"] != name]


def span(rs: Rows, col: str, digits: int = 0, scale: float = 100) -> tuple[str, str]:
    v = [scale * num(r, col) for r in rs]
    return f"{min(v):.{digits}f}", f"{max(v):.{digits}f}"


def power_span(rs: Rows, col: str) -> tuple[str, str, str]:
    """Range of a power over the seven stocks other than RRC, then RRC's value (percent)."""
    return (*span(others(rs), col), pct(num(stock(rs, "RRC"), col), 0))


def power_vs_t_test(rs: Rows) -> tuple[str, str, str]:
    """The largest gap (points, rounded up) between the simulated power at 2% and the t-test
    cross-check over the stocks other than MRK, and MRK's two values."""
    gap = max(abs(100 * (num(r, "power_bmp_2pct") - num(r, "plugin_t_power_2pct")))
              for r in others(rs, "MRK"))
    m = stock(rs, "MRK")
    return (str(math.ceil(gap)), pct(num(m, "plugin_t_power_2pct"), 0),
            pct(num(m, "power_bmp_2pct"), 0))


def largest(rs: Rows, asset: str) -> Rows:
    return [r for r in rs if r["asset"] == asset]


def flag_only(rs: Rows) -> Rows:
    return [r for r in rs if r["in"] == "flag"]


def below_64(r: dict[str, str]) -> bool:
    return r["us_geometric_max_vmax_kt"] != "" and num(r, "us_geometric_max_vmax_kt") < 64


def flag_only_counts(rs: Rows) -> tuple[int, int, int]:
    """Flag-only storms with an interpolated wind below 64 kt, of which flagged at 65 kt, and
    the flagged wind of Dorian 2019."""
    f = flag_only(rs)
    return (sum(below_64(x) for x in f),
            sum(below_64(x) and x["us_flag_max_vmax_kt"] == "65.0" for x in f),
            int(num(where(rs, name="DORIAN", season="2019"), "us_flag_max_vmax_kt")))


def three_interpolated(rs: Rows) -> tuple[str, str]:
    v = [num(x, "us_geometric_max_vmax_kt") for x in rs
         if (x["name"], x["season"]) in THREE_FLAG_ONLY]
    return f"{min(v):.0f}", f"{max(v):.0f}"


def n_below(rs: Rows, cols: tuple[str, ...], level: float = 0.05) -> int:
    return sum(num(x, c) < level for x in rs for c in cols)


def includes_zero(r: dict[str, str]) -> bool:
    return num(r, "gross_lo") < 0 < num(r, "gross_hi")


# (file, function of its rows giving the numbers, README sentence with {} for each of them)
DERIVED: list[tuple[str, Callable[[Rows], tuple | str], str]] = [
    # Real-data event study: smallest p-values, sensitivity analyses, pooled test.
    (f"{RE}/car_tests_primary.csv", lambda r: (f"{min_p(r):.2f}", f"{min_p(r, ('bmp_p',)):.2f}"),
     "the smallest raw p-value of any test is {}, of the BMP test {})"),
    (f"{RE}/sensitivity_car_tests.csv", lambda r: (WORDS[n_below(r, P4)], 4 * len(r)),
     "give {} Patell p-values below 0.05 out of {}"),
    (f"{RE}/pooled_primary.csv",
     lambda r: where(r, group="storm (signed)", window="post[0,+10]")["n"], "({} asset-events;"),
    # Per-stock power: the introduction, the reading of section 1 and the limitations.
    (f"{PS}/summary.csv", lambda r: power_span(r, "power_bmp_2pct"),
     "detect a true 2% abnormal return {}-{}% of the time for seven of the stocks and {}% for the "
     "eighth (RRC)"),
    (f"{PS}/summary.csv", lambda r: power_span(r, "power_bmp_2pct"),
     "a true 11-day CAR of 2% is detected {}-{}% of the time for seven stocks and {}% for RRC"),
    (f"{PS}/summary.csv",
     lambda r: (pct(num(stock(r, "RRC"), "resid_sd_rms"), 1),
                *span(others(r), "resid_sd_rms", digits=1)),
     "residual volatility ({}% a day) is more than twice the others' ({}-{}%)"),
    (f"{PS}/summary.csv", lambda r: power_span(r, "power_bmp_1pct"),
     "a true 1% is detected {}-{}% of the time (RRC {}%)"),
    (f"{PS}/summary.csv",
     lambda r: (*span(others(r), "delta_80pct_power_bmp", digits=1),
                pct(num(stock(r, "RRC"), "delta_80pct_power_bmp"), 1)),
     "the CAR detected with 80% power is {}-{}% (RRC {}%)"),
    (f"{PS}/summary.csv", power_vs_t_test,
     "it is within {} points of the simulation except for MRK ({}% against {}%)"),
    (f"{PS}/summary.csv", lambda r: pct(num(stock(r, "MRK"), "real_car_sd"), 1),
     "whose standard deviation ({}%) is the largest of the seven"),
    (f"{PS}/summary.csv",
     lambda r: (min(int(x["events"]) for x in r), max(int(x["events"]) for x in r)),
     "Per-stock tests have {}-{} events"),
    (f"{PS}/summary.csv", lambda r: power_span(r, "power_bmp_2pct"),
     "and detect a true CAR of 2% {}-{}% of the time (RRC {}%)"),
    # MRK's three largest CARs.
    (f"{RE}/largest_events.csv",
     lambda r: tuple(f"{100 * num(x, 'car_0_10'):+.0f}" for x in largest(r, "MRK")),
     "({}% Ivan 2004, {}% Dolly 2008, {}% Wilma 2005;"),
    # Trading: R2 (the forecast-trigger rule) and R1 (the calendar rule).
    (f"{RE}/trading_forward.csv", lambda r: r2(r)["events"], "(forward-only, {} events)"),
    (f"{RE}/trading_forward.csv", lambda r: pct_ci(r2(r), "mean_net", "boot_lo", "boot_hi"),
     "and {} when each season's configuration is chosen on earlier seasons only"),
    (f"{RE}/trading_loso.csv", lambda r: pct_ci(r2(r), "mean_gross", "gross_lo", "gross_hi"),
     "Before costs the same events give {} and"),
    (f"{RE}/trading_forward.csv", lambda r: pct_ci(r2(r), "mean_gross", "gross_lo", "gross_hi"),
     "and {}, which cannot be told from zero"),
    (f"{RE}/trading_loso.csv",
     lambda r: pct(num(r2(r), "mean_gross") - num(r2(r), "mean_net"), 1),
     "the difference of about {}% per event is the assumed costs"),
    (f"{RE}/trading_loso.csv",
     lambda r: (pct(num(r1(r), "mean_net")), pct(num(r1(r), "mean_gross"))),
     "not better ({}% net, {}% before costs, leave-one-season-out)"),
    # Landfall definitions: the storms found only by the flags.
    (f"{LC}/storm_level_disagreements.csv", flag_only_counts,
     "{} have a geometric US crossing whose interpolated wind is below 64 kt ({} of them are "
     "flagged at 65 kt) and the 22nd is Dorian 2019, flagged at {} kt"),
    (f"{LC}/storm_level_disagreements.csv", three_interpolated,
     "Gaston 2004 (flagged at 65 kt, {}-{} kt interpolated)"),
    # The own-prices pipeline on synthetic prices.
    (f"{OP}/findings_event_study.csv",
     lambda r: (WORDS[n_below(r, ("placebo_p",))], len(r)),
     "but {} of the {} placebo p-values (13 assets and the pooled storm group) are below 0.05"),
]

# (file, claim about its rows, README text): the claim must hold and the text must occur.
FACTS: list[tuple[str, Callable[[Rows], bool], str]] = [
    (f"{RE}/car_tests_primary.csv",
     lambda r: len(r) == 24 and all(num(x, "bmp_p_holm") == 1 for x in r),
     "every per-stock Holm-adjusted p-value is 1"),
    (f"{RE}/car_tests_primary.csv", lambda r: min_p(r) >= 0.05,
     "No stock has a significant CAR by any of the four per-stock tests"),
    (f"{RE}/sensitivity_car_tests.csv",
     lambda r: n_below(r, ("bmp_p", "naive_p", "rank_p")) == 0,
     "none from the BMP, naive or rank tests"),
    (f"{RE}/sensitivity_car_tests.csv",
     lambda r: sorted((x["asset"], x["sensitivity"]) for x in r if num(x, "patell_p") < 0.05)
     == sorted([SENSITIVITY_FLAGS, SENSITIVITY_ALL, SENSITIVITY_ALL]),
     "(RRC with the flagged events, MRK with all events, twice)"),
    (f"{RE}/findings_event_study.csv",
     lambda r: where(r, series="RRC")["sign_as_hypothesised"] == "False",
     "RRC's sign is opposite to its hypothesis"),
    (f"{RE}/trading_loso.csv", lambda r: includes_zero(r2(r)), "which cannot be told from zero"),
    (f"{RE}/trading_forward.csv",
     lambda r: includes_zero(r2(r)) and num(r2(r), "boot_hi") < 0,
     "loses money after assumed costs and not measurably before them"),
    (f"{PS}/summary.csv",
     lambda r: num(stock(r, "RRC"), "resid_sd_rms") > 2 * max(num(x, "resid_sd_rms")
                                                              for x in others(r)),
     "is more than twice the others'"),
    (f"{PS}/summary.csv",
     lambda r: num(stock(r, "MRK"), "real_car_sd") == max(num(x, "real_car_sd")
                                                         for x in others(r)),
     "is the largest of the seven"),
    (f"{RE}/largest_events.csv",
     lambda r: len(largest(r, "MRK")) == 3
     and all(abs(num(x, "car_0_10")) > 0.10 for x in largest(r, "MRK")),
     "three of its CARs are beyond 10% in size"),
    (f"{LC}/storm_level_disagreements.csv",
     lambda r: len(flag_only(r)) == 22
     and where(r, name="DORIAN", season="2019")["us_geometric_max_vmax_kt"] == "",
     "with no geometric US crossing at all"),
]

INTERVAL = re.compile(r"-?\d+(?:\.\d+)?%? \[-?\d+(?:\.\d+)?, -?\d+(?:\.\d+)?\]")
TABLE = re.compile(r"<!-- table:(\w+) -->\n(.*?)<!-- /table:\1 -->", re.DOTALL)


def generated_tables(text: str) -> dict[str, str]:
    return {k: v for k, v in TABLE.findall(text)}


def broken_links(readme: str) -> list[str]:
    missing = []
    for target in re.findall(r"\]\(([^)\s]+)\)", readme):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        if not (ROOT / target.split("#", 1)[0]).exists():
            missing.append(target)
    return missing


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tables_md = (ROOT / "results" / "tables.md").read_text(encoding="utf-8")
    problems = 0

    in_readme = generated_tables(readme)
    for key, body in in_readme.items():
        if body.strip() not in tables_md:
            print(f"README table {key!r} differs from results/tables.md (run make_tables.py)")
            problems += 1
    prose = re.sub(r"\s+", " ", TABLE.sub("", readme))  # line breaks do not matter

    for target in broken_links(readme):
        print(f"README.md links to {target!r}, which does not exist")
        problems += 1
    texts: dict[str, str] = {}
    for path, in_file, in_text in CHECKS:
        if path not in texts:
            try:
                texts[path] = (ROOT / path).read_text(encoding="utf-8")
            except FileNotFoundError:
                print(f"{path}: file not found")
                problems += 1
                continue
        if in_file not in texts[path]:
            print(f"{path}: {in_file!r} not found")
            problems += 1
        if in_text not in prose:
            print(f"README.md prose does not contain {in_text!r} (checked against {path})")
            problems += 1
    for path, derive, template in DERIVED:
        try:
            value = derive(rows(path))
        except (FileNotFoundError, ValueError, KeyError, IndexError) as err:
            print(f"{path}: cannot compute the numbers of {template!r}: {err!r}")
            problems += 1
            continue
        text = template.format(*(value if isinstance(value, tuple) else (value,)))
        if text not in prose:
            print(f"README.md prose does not contain {text!r} (computed from {path})")
            problems += 1
    for path, holds, in_text in FACTS:
        try:
            ok = holds(rows(path))
        except (FileNotFoundError, ValueError, KeyError, IndexError) as err:
            print(f"{path}: cannot evaluate the claim {in_text!r}: {err!r}")
            problems += 1
            continue
        if not ok:
            print(f"{path}: the claim behind {in_text!r} does not hold")
            problems += 1
        if in_text not in prose:
            print(f"README.md prose does not contain {in_text!r} (claim about {path})")
            problems += 1
    quoted = sorted(set(INTERVAL.findall(prose)))
    for q in quoted:
        if q.replace("%", "") not in tables_md:
            print(f"README.md quotes {q!r}, which is in no table of results/tables.md")
            problems += 1
    n_checks = len(CHECKS) + len(DERIVED) + len(FACTS)
    print(f"{len(in_readme)} generated tables, {n_checks} quoted numbers and claims, "
          f"{len(quoted)} intervals in prose: {problems} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
