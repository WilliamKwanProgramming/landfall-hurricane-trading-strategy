#!/usr/bin/env python3
"""Run docs/protocol_own_prices.md on prices you supply: event study, placebo, multiple testing,
the R1 and R2 trading rules and the deflated Sharpe ratio, for the assets of an assets.csv.

Inputs: a price CSV (schema in landfall/prices.py; it is read, never copied) and assets.csv
(template docs/own_prices_assets.csv). The events, the placebo exclusions and the forecast
probabilities are read from results/ (committed wave-1 outputs computed from hurricane tracks
only). Writes CSV files and a MANIFEST with the SHA-256 of both inputs to --out; no price is
written. Every output row carries the --label ("real" or "synthetic").

Usage:
  python scripts/run_own_prices.py --prices PATH --assets assets.csv --market SPX --label real \
      [--out results/own_prices] [--specifications 1] [--seed 20261105] [--placebo-sets 1000]
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import time

import pandas as pd

from landfall import pipeline as pl
from landfall import prices as P
from landfall.provenance import REPO, write_manifest
from landfall.study import backtest as bt

SEASONS = (1990, 2022)
EVENTS = REPO / "results" / "real_eventstudy" / "events.csv"
EXCLUSIONS = REPO / "results" / "landfall_counts" / "us_landfalls_any_intensity_1985_2024.csv"
PROBABILITIES = REPO / "results" / "real_eventstudy" / "landfall_probabilities.csv.gz"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prices", required=True, type=pathlib.Path)
    ap.add_argument("--assets", required=True, type=pathlib.Path)
    ap.add_argument("--market", default="SPX", help="column of the market index (S&P 500)")
    ap.add_argument("--label", required=True, choices=["real", "synthetic"],
                    help="written into every output: are these prices real or simulated?")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results" / "own_prices")
    ap.add_argument("--specifications", type=int, default=1,
                    help="number of specifications run so far, this one included (DSR uses "
                         "N = 9 x this)")
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--placebo-sets", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.specifications < 1:
        ap.error("--specifications must be >= 1")
    t_start = time.monotonic()

    assets = pl.load_assets(args.assets)
    prices = P.load_prices(args.prices, args.market)
    missing = [a.name for a in assets if a.name not in prices.columns]
    if missing:
        raise SystemExit(f"{args.prices}: no column for {missing}. Drop them from assets.csv "
                         "and from your protocol copy (before loading prices), or fix the file.")
    unused = sorted(set(prices.columns) - {a.name for a in assets} - {args.market})
    prices = prices[[a.name for a in assets] + [args.market]]
    checks = P.check_prices(prices, args.market)
    datas = []
    for a in assets:
        r = P.asset_returns(prices, a.name, args.market)
        datas.append(pl.AssetData(a, r.dates, r.ret, r.mkt))
    m = prices[args.market].dropna()
    calendar = pd.DatetimeIndex(m.index[1:])

    events = pd.read_csv(EVENTS, parse_dates=["time"])
    prim = events[events["kept_primary"]].sort_values("time").reset_index(drop=True)
    excl = pd.read_csv(EXCLUSIONS, parse_dates=["time"])
    probs = pd.read_csv(PROBABILITIES, parse_dates=["time", "available_time"])
    n_years = SEASONS[1] - SEASONS[0] + 1

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    def save(df: pd.DataFrame, name: str) -> None:
        df = df.copy()
        df.insert(0, "data", args.label)
        df.to_csv(out / name, index=False, float_format="%.6g")

    save(checks, "data_checks.csv")
    per_asset, pooled, _, cov = pl.event_study(datas, prim["time"], args.seed)
    save(cov, "coverage.csv")
    save(per_asset, "car_tests.csv")
    save(pooled, "pooled.csv")
    plc, plc_signed = pl.placebo(datas, calendar, prim["time"], excl["time"],
                                 (SEASONS[0] + 1, SEASONS[1]), args.placebo_sets, args.seed)
    prow = pl.placebo_table(per_asset, pooled, plc, plc_signed)
    save(prow, "placebo.csv")
    save(pl.findings(assets, per_asset, pooled, prow), "findings_event_study.csv")

    costs = bt.Costs()
    rules = (("R1 calendar (not tradeable)", pl.R1_GRID, pl.calendar_entries(prim),
              pl.R1_PRIMARY),
             ("R2 forecast trigger", pl.R2_GRID, pl.trigger_entries(probs), pl.R2_PRIMARY))
    tabs, dsrs, losos = [], [], []
    for name, grid, fn, primary in rules:
        t, d, lrow, _ = pl.evaluate_rule(name, grid, fn, datas, costs, n_years, args.seed,
                                         primary, trials_multiplier=args.specifications)
        tabs.append(t)
        dsrs.append(d)
        losos.append(lrow)
    save(pd.concat(tabs), "trading_configs.csv")
    save(pd.DataFrame(dsrs), "trading_dsr.csv")
    save(pd.DataFrame(losos), "trading_loso.csv")
    save(pd.DataFrame([pl.trading_finding(dsrs[1], losos[1])]), "findings_trading.csv")

    all_ev = events.sort_values("time").reset_index(drop=True)
    pa, po, _, _ = pl.event_study(datas, all_ev["time"], args.seed)
    pa = pa.drop(columns=["bmp_p_holm", "bmp_p_bh"])  # adjusted only in the primary family
    save(pa.assign(sensitivity="all events kept"), "sensitivity_car_tests.csv")
    save(po.assign(sensitivity="all events kept"), "sensitivity_pooled.csv")
    double = bt.Costs(20.0, 4.0, 100.0)
    crow = []
    for name, _, fn, primary in rules:
        evt = pl.event_table(pl.trades_for(datas, primary, fn, double))
        crow.append({"rule": name, "config": str(primary), "costs": "doubled",
                     **pl.metrics(evt, n_years, args.seed)})
    save(pd.DataFrame(crow), "sensitivity_costs.csv")

    elapsed = time.monotonic() - t_start
    write_manifest(out, args.seed, {
        "data": f"{args.label.upper()}: prices supplied by the user (not stored here)",
        "protocol": "docs/protocol_own_prices.md",
        "prices_file": args.prices.name, "prices_sha256": sha256(args.prices),
        "assets_file": args.assets.name, "assets_sha256": sha256(args.assets),
        "market_column": args.market,
        "assets": ", ".join(f"{a.name} ({a.group}, {a.kind}, sign {a.sign:+d})"
                            for a in assets),
        "columns_in_file_not_used": ", ".join(unused) or "none",
        "events_primary": len(prim), "events_all": len(events),
        "placebo_sets": args.placebo_sets, "specifications": args.specifications,
        "dsr_trials": 9 * args.specifications,
        "elapsed_seconds": f"{elapsed:.0f}"})
    find = pd.read_csv(out / "findings_event_study.csv")
    print(f"[{args.label}] wrote {out} in {elapsed:.0f} s")
    print(per_asset[per_asset.window == "post[0,+10]"][
        ["asset", "group", "n", "mean_car", "boot_lo", "boot_hi", "bmp_p", "bmp_p_holm"]]
        .to_string(index=False))
    print(find.to_string(index=False))
    print(pd.DataFrame(losos)[["rule", "events", "mean_net", "boot_lo", "boot_hi"]]
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
