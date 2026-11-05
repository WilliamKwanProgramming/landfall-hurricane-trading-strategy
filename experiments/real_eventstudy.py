#!/usr/bin/env python3
"""Experiment (d): the pre-specified real-data event study of docs/protocol.md.

Real data: skfolio daily prices of 20 US stocks and the S&P 500 index (1990-2022) and HURDAT2
best tracks. Run once, after the protocol was committed. Writes results/real_eventstudy/.

Usage: python experiments/real_eventstudy.py [--seed 20261105] [--placebo-sets 1000]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from landfall import events as ev
from landfall import forecast as F
from landfall import pipeline as pl
from landfall import realstudy as rs
from landfall import signals
from landfall.data import hurdat2, paths
from landfall.data.prices import load_skfolio_prices, simple_returns
from landfall.data.tracks import load_combined_tracks
from landfall.geo import atlantic_land_grid
from landfall.provenance import results_dir, write_manifest
from landfall.study import backtest as bt
from landfall.study import eventstudy as es

STORM = {"HD": 1, "XOM": -1, "CVX": -1, "RRC": 1}
CONTROLS = ["MSFT", "JNJ", "MRK", "PFE"]
SEASONS = (1990, 2022)
R1_GRID, R2_GRID, R1_PRIMARY, R2_PRIMARY = pl.R1_GRID, pl.R2_GRID, pl.R1_PRIMARY, pl.R2_PRIMARY


def asset_data(ret: pd.DataFrame, mkt: np.ndarray) -> list[pl.AssetData]:
    """The eight stocks of the protocol, all on the common skfolio calendar."""
    assets = [pl.Asset(a, "storm", s) for a, s in STORM.items()]
    assets += [pl.Asset(a, "control", 0) for a in CONTROLS]
    return [pl.AssetData(a, ret.index, ret[a.name].to_numpy(), mkt) for a in assets]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--placebo-sets", type=int, default=1000)
    args = ap.parse_args()
    out = results_dir("real_eventstudy")

    prices, index = load_skfolio_prices()
    ret = simple_returns(prices)
    mkt = simple_returns(index).to_numpy()
    dates = ret.index

    grid = atlantic_land_grid()
    tracks, _ = load_combined_tracks()
    geo = ev.geometric_landfalls(tracks[tracks.season >= 1985], grid)
    events = ev.storms_with_us_hurricane_landfall(geo)
    events = events[events.season.between(*SEASONS)].sort_values("time").reset_index(drop=True)
    us_all = ev.us_landfalls(geo)  # any intensity, for the placebo exclusion
    day0 = es.event_rows(dates, events["time"])
    fits_ok = np.array([rs.window_fits(int(d), len(dates)) for d in day0])
    keep = np.zeros(len(events), dtype=bool)
    keep[fits_ok] = rs.drop_overlapping(day0[fits_ok])
    events["day0_date"] = [dates[d].date() if d >= 0 else None for d in day0]
    events["windows_fit"] = fits_ok
    events["kept_primary"] = keep
    events.to_csv(out / "events.csv", index=False)
    n_years = SEASONS[1] - SEASONS[0] + 1

    # Primary event study.
    prim = events[keep].reset_index(drop=True)
    datas = asset_data(ret, mkt)
    per_asset, pooled, _, _ = pl.event_study(datas, prim["time"], args.seed)
    per_asset.to_csv(out / "car_tests_primary.csv", index=False, float_format="%.6g")
    pooled.to_csv(out / "pooled_primary.csv", index=False, float_format="%.6g")

    # Placebo.
    plc, plc_signed = pl.placebo(datas, dates, prim["time"], us_all["time"],
                                 (SEASONS[0] + 1, SEASONS[1]), args.placebo_sets, args.seed)
    prow = pl.placebo_table(per_asset, pooled, plc, plc_signed)
    prow.to_csv(out / "placebo.csv", index=False, float_format="%.6g")

    # Findings by the protocol's criteria.
    assets = [d.asset for d in datas]
    pl.findings(assets, per_asset, pooled, prow).to_csv(
        out / "findings_event_study.csv", index=False, float_format="%.6g")

    # Trading rules.
    costs = bt.Costs()

    r1_entries = pl.calendar_entries(prim)

    samples = F.build_samples(tracks)
    probs = signals.landfall_probabilities(samples, range(SEASONS[0], SEASONS[1] + 1),
                                           atlantic_land_grid(coarsen=6), 200, args.seed)
    probs.to_csv(out / "landfall_probabilities.csv.gz", index=False, float_format="%.4f",
                 compression={"method": "gzip", "mtime": 0})

    r2_entries = pl.trigger_entries(probs)

    tabs, dsrs, losos, fwds = [], [], [], []
    for name, grid_, fn, prim_cfg in (("R1 calendar (not tradeable)", R1_GRID, r1_entries,
                                       R1_PRIMARY),
                                      ("R2 forecast trigger", R2_GRID, r2_entries, R2_PRIMARY)):
        t, d, lrow, _, frow = pl.evaluate_rule(name, grid_, fn, datas, costs, n_years, args.seed,
                                               prim_cfg)
        tabs.append(t)
        dsrs.append(d)
        losos.append(lrow)
        fwds.append(frow)
    pd.concat(tabs).to_csv(out / "trading_configs.csv", index=False, float_format="%.6g")
    pd.DataFrame(dsrs).to_csv(out / "trading_dsr.csv", index=False, float_format="%.6g")
    pd.DataFrame(losos).to_csv(out / "trading_loso.csv", index=False, float_format="%.6g")
    # Added after an independent review of the first run (docs/protocol.md, Deviations): the
    # forward-only (expanding-window) evaluation; gross-of-cost columns are in the files above.
    pd.DataFrame(fwds).to_csv(out / "trading_forward.csv", index=False, float_format="%.6g")
    pd.DataFrame([pl.trading_finding(dsrs[1], losos[1])]).to_csv(
        out / "findings_trading.csv", index=False)

    # Sensitivities.
    sens = []
    pa, po, _, _ = pl.event_study(datas, events[fits_ok]["time"], args.seed)
    pa = pa.drop(columns=["bmp_p_holm", "bmp_p_bh"])  # adjusted only in the primary family
    pa["sensitivity"], po["sensitivity"] = "all events kept", "all events kept"
    sens.append((pa, po))
    rda, _ = hurdat2.load_rda_mirror(paths.raw_file(paths.RDA_MIRROR))
    flag_ev = ev.storms_with_us_hurricane_landfall(ev.flagged_landfalls(rda))
    flag_ev = flag_ev[flag_ev.season.between(1990, 2019)].sort_values("time")
    fd0 = es.event_rows(dates, flag_ev["time"])
    fok = np.array([rs.window_fits(int(d), len(dates)) for d in fd0])
    fkeep = rs.drop_overlapping(fd0[fok])
    pa, po, _, _ = pl.event_study(datas, flag_ev[fok][fkeep]["time"], args.seed)
    pa = pa.drop(columns=["bmp_p_holm", "bmp_p_bh"])
    pa["sensitivity"], po["sensitivity"] = "flag events 1990-2019", "flag events 1990-2019"
    sens.append((pa, po))
    pd.concat([s[0] for s in sens]).to_csv(out / "sensitivity_car_tests.csv", index=False,
                                           float_format="%.6g")
    pd.concat([s[1] for s in sens]).to_csv(out / "sensitivity_pooled.csv", index=False,
                                           float_format="%.6g")
    double = bt.Costs(20.0, 4.0, 100.0)
    crow = []
    for name, cfg, fn in (("R1 calendar (not tradeable)", R1_PRIMARY, r1_entries),
                          ("R2 forecast trigger", R2_PRIMARY, r2_entries)):
        evt = pl.event_table(pl.trades_for(datas, cfg, fn, double))
        crow.append({"rule": name, "config": str(cfg), "costs": "doubled",
                     **pl.metrics(evt, n_years, args.seed)})
    pd.DataFrame(crow).to_csv(out / "sensitivity_costs.csv", index=False, float_format="%.6g")

    write_manifest(out, args.seed, {
        "data": "REAL: skfolio prices 1990-2022, HURDAT2 tracks (GitHub mirrors)",
        "protocol": "docs/protocol.md", "placebo_sets": args.placebo_sets,
        "events_total": len(events), "events_primary": int(keep.sum()),
        "events_dropped_overlap": int(fits_ok.sum() - keep.sum()),
        "events_outside_price_data": int((~fits_ok).sum())})
    print(per_asset[per_asset.window == "post[0,+10]"].to_string())
    print(pooled.to_string())
    print(prow.to_string())
    print(pd.concat(tabs).to_string())
    print(pd.DataFrame(dsrs).to_string())
    print(pd.DataFrame(losos).to_string())
    print(pd.DataFrame(fwds).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
