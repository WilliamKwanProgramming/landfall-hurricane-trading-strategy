#!/usr/bin/env python3
"""Check that the numbers in README.md are the numbers in the committed results files.

Three checks:
1. Every generated table between <!-- table:KEY --> markers in README.md is identical to the
   table KEY in results/tables.md (written by scripts/make_tables.py).
2. Each entry of CHECKS names a results file, a text that must occur in it and a text that must
   occur in README.md (outside the generated tables: numbers quoted in prose). When a result is
   regenerated and its number changes, the first part fails; when the README is edited by hand
   and no longer quotes the number, the second fails.
3. Every interval written in the README prose as "x [lo, hi]" occurs in results/tables.md, and
   every relative link or image points to a file that exists.

Usage: python scripts/check_readme_numbers.py      (exit status 1 on any problem)
Standard library only.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RE = "results/real_eventstudy"
SY = "results/synthetic_eventstudy"
LC = "results/landfall_counts"
BM = "results/benchmark"

CHECKS = [
    # Real-data event study: the pooled signed test, the forecast-trigger rule, the placebo.
    (f"{RE}/MANIFEST.txt", "events_primary: 39", "39 US hurricane landfalls"),
    (f"{RE}/findings_event_study.csv", "HD,1,", "Holm-adjusted p-values of 1.0"),
    (f"{RE}/placebo.csv", '"storm (signed, pooled)",-0.00229995,', "-0.23%"),
    (f"{RE}/placebo.csv", ",0.478,0.956", "placebo p = 0.96"),
    (f"{RE}/trading_loso.csv", "R2 forecast trigger,152,-0.00351724,-0.00629465,-0.000960345",
     "-0.35% [-0.63, -0.10]"),
    (f"{RE}/trading_dsr.csv", ",148,0.0124617", "deflated Sharpe ratio of 0.01"),
    # Power of the design (synthetic GARCH-t simulations).
    (f"{SY}/garch_t.csv", ",0.40500,0.01098,", "about 40%"),
    (f"{SY}/garch_t.csv", ",0.89600,0.00683,", "about 90%"),
    (f"{SY}/garch_t_35x1.csv", ",0.15550,0.00810,", "about 16%"),
    (f"{SY}/garch_t_35x1.csv", ",0.43950,0.01110,", "about 44%"),
    # Landfall definitions: facts about the storms found by one definition only.
    (f"{LC}/storm_level_disagreements.csv", "AL041970,geometric,CELIA,1970,0,", "Celia 1970"),
    (f"{LC}/storm_level_agreement_1851_2019.csv", "geometric_only,16", "12 of the 16"),
    (f"{LC}/storm_level_agreement_1851_2019.csv", "flag_only,22", "14 of the 22"),
    # Footprint kernel benchmark (indicative timings; they change when the benchmark is rerun).
    (f"{BM}/footprint_speed.csv", "ratio numpy / cpp optimised,15,2.2,1.1,4.2",
     "2.2 times faster than NumPy"),
    (f"{BM}/footprint_speed.csv", "ratio numpy / cpp optimised,15,2.2,1.1,4.2", "1.1 to 4.2"),
    (f"{BM}/footprint_speed.csv", "ratio numpy / cpp reference,15,0.9,", "ratio 0.9"),
    (f"{BM}/wave1/footprint_speed.csv", "ratio numpy/cpp,15,1.1,0.7,1.5", "ratio of medians 1.1"),
    (f"{BM}/MANIFEST.txt", "load1_before: 1.57", "from 1.57 to 2.18"),
    (f"{BM}/MANIFEST.txt", "load1_after: 2.18", "from 1.57 to 2.18"),
    (f"{BM}/MANIFEST.txt", "process_niceness: 10", "at nice 10"),
    (f"{BM}/MANIFEST.txt", "cpp optimised: 2.94e-14", "2.94e-14"),
    (f"{BM}/callgrind.csv", "reference,241000,1044,0.408,0.153,", "1,044 instructions"),
    (f"{BM}/callgrind.csv", "reference,241000,1044,0.408,0.153,", "41% of them in sine"),
    (f"{BM}/callgrind.csv", "reference,241000,1044,0.408,0.153,", "15% in `hypot`"),
    (f"{BM}/callgrind.csv", "optimised,241000,376,0.022,0.003,0.341,", "376 instructions"),
    (f"{BM}/callgrind.csv", "optimised,241000,376,0.022,0.003,0.341,", "34% of them in `pow`"),
    (f"{LC}/definitions.csv", '"US coast, vmax >= 64 kt, 1851-2019",333,336,', "333 flagged, 336"),
    # Own-prices pipeline on synthetic prices and the mutation check.
    ("results/placebo_size/summary.csv", "placebo,0.05,240,0.158,0.024",
     "15.8% of tests (Monte Carlo SE 2.4)"),
    ("results/placebo_size/summary.csv", "bmp,0.05,240,0.05,0.014", "BMP holds 5.0%"),
]

INTERVAL = re.compile(r"-?\d+(?:\.\d+)?%? \[-?\d+(?:\.\d+)?, -?\d+(?:\.\d+)?\]")
TABLE = re.compile(r"<!-- table:(\w+) -->\n(.*?)<!-- /table:\1 -->", re.S)


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
    quoted = sorted(set(INTERVAL.findall(prose)))
    for q in quoted:
        if q.replace("%", "") not in tables_md:
            print(f"README.md quotes {q!r}, which is in no table of results/tables.md")
            problems += 1
    print(f"{len(in_readme)} generated tables, {len(CHECKS)} quoted numbers, "
          f"{len(quoted)} intervals in prose: {problems} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
