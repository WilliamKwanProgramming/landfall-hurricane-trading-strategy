#!/usr/bin/env python3
"""Experiment (c): the two landfall definitions compared, US hurricane landfalls per season, and
a Poisson dispersion check. Also writes the comparison of the two HURDAT2 mirrors.

Real data: HURDAT2 best tracks (GitHub mirrors, see scripts/fetch_data.py). Deterministic.
Writes results/landfall_counts/ and results/data_checks/.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from landfall import events as ev
from landfall.data import hurdat2, paths
from landfall.data.tracks import load_combined_tracks
from landfall.geo import atlantic_land_grid
from landfall.provenance import results_dir, write_manifest

ERAS = [(1851, 1899), (1900, 1949), (1950, 1989), (1990, 2019)]


def compare(flag: pd.DataFrame, geo: pd.DataFrame, label: str) -> dict:
    m = ev.match_definitions(flag, geo)
    matched = m["geo_index"].notna()
    return {"subset": label, "flag": len(flag), "geometric": len(geo),
            "flag_matched": int(matched.sum()),
            "share_flag_matched": float(matched.mean()) if len(m) else np.nan,
            "geometric_unmatched": int(len(geo) - matched.sum()),
            "median_hours_apart": float(m.loc[matched, "hours_apart"].median()),
            "median_km_apart": float(m.loc[matched, "km_apart"].median())}


def per_season(storms: pd.DataFrame, first: int, last: int) -> pd.Series:
    s = storms[(storms.season >= first) & (storms.season <= last)]
    return s.groupby("season").size().reindex(range(first, last + 1), fill_value=0)


def main() -> int:
    grid = atlantic_land_grid()
    rda, rda_notes = hurdat2.load_rda_mirror(paths.raw_file(paths.RDA_MIRROR))
    dpl, dpl_notes = hurdat2.load_dplyr_mirror(paths.raw_file(paths.DPLYR_MIRROR))

    # ---- mirror comparison ----------------------------------------------------------------
    dc = results_dir("data_checks")
    cmp = hurdat2.compare_sources(rda, dpl)
    diffs = pd.DataFrame(cmp.pop("differences"), columns=["storm", "time", "field", "rda",
                                                          "dplyr"])
    diffs.to_csv(dc / "mirror_differences.csv", index=False)
    cmp["storms_only_rda"] = len(cmp["storms_only_rda"])
    cmp["storms_only_dplyr"] = len(cmp["storms_only_dplyr"])
    cmp["rda_loader_notes"] = rda_notes
    cmp["dplyr_loader_notes"] = dpl_notes
    (dc / "mirror_comparison.json").write_text(json.dumps(cmp, indent=1) + "\n")
    write_manifest(dc, None)

    # ---- definitions on 1851-2019 ------------------------------------------------------------
    out = results_dir("landfall_counts")
    flag = ev.flagged_landfalls(rda)
    geo = ev.geometric_landfalls(rda, grid)
    rows = [compare(flag, geo, "all landfalls 1851-2019")]
    for lo, hi in ERAS:
        rows.append(compare(flag[flag.season.between(lo, hi)], geo[geo.season.between(lo, hi)],
                            f"all landfalls {lo}-{hi}"))
    fu, gu = ev.us_landfalls(flag), ev.us_landfalls(geo)
    rows.append(compare(fu, gu, "US coast landfalls 1851-2019"))
    rows.append(compare(ev.us_landfalls(flag, 64.0), ev.us_landfalls(geo, 64.0),
                        "US coast, vmax >= 64 kt, 1851-2019"))
    pd.DataFrame(rows).to_csv(out / "definitions.csv", index=False)

    sf = ev.storms_with_us_hurricane_landfall(flag)
    sg = ev.storms_with_us_hurricane_landfall(geo)
    a, b = set(sf.storm_id), set(sg.storm_id)
    storm_level = {"storms_with_US_hurricane_landfall_flag": len(a),
                   "storms_with_US_hurricane_landfall_geometric": len(b),
                   "both": len(a & b), "flag_only": len(a - b), "geometric_only": len(b - a)}
    pd.Series(storm_level).to_csv(out / "storm_level_agreement_1851_2019.csv", header=["count"])
    pd.DataFrame({"storm_id": sorted(a ^ b),
                  "in": ["flag" if s in a else "geometric" for s in sorted(a ^ b)]}).to_csv(
        out / "storm_level_disagreements.csv", index=False)

    # ---- per-season counts and dispersion ------------------------------------------------------
    comb, _ = load_combined_tracks()
    geo_comb = ev.geometric_landfalls(comb[comb.season >= 1985], grid)
    sgc = ev.storms_with_us_hurricane_landfall(geo_comb)
    series = {
        "flag_1900_2019": per_season(sf, 1900, 2019),
        "geometric_1900_2019": per_season(sg, 1900, 2019),
        "flag_1990_2019": per_season(sf, 1990, 2019),
        "geometric_1990_2022": per_season(sgc, 1990, 2022),
    }
    pd.DataFrame(series).to_csv(out / "us_hurricane_storms_per_season.csv",
                                index_label="season")
    disp = []
    for k, s in series.items():
        r = ev.dispersion_test(s.to_numpy())
        r["series"] = k
        disp.append(r)
    pd.DataFrame(disp)[["series", "n_seasons", "mean", "variance", "ratio", "D", "df",
                        "p_two_sided"]].to_csv(out / "dispersion.csv", index=False)
    # Event list used downstream (geometric definition, combined tracks, 1990-2022).
    sgc[sgc.season.between(1990, 2022)].to_csv(out / "us_hurricane_landfalls_1990_2022.csv",
                                                 index=False)
    write_manifest(out, None, {"land_mask": "global-land-mask 1.0.0, 1/120 degree, Atlantic crop",
                               "geometric_rule": "10-min steps, >= 1 h at sea before crossing",
                               "matching": "same storm, <= 6 h and <= 150 km"})
    print(pd.DataFrame(rows).to_string())
    print(storm_level)
    print(pd.DataFrame(disp).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
