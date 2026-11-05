#!/usr/bin/env python3
"""Write a SYNTHETIC price file for the assets of an assets.csv, to try scripts/run_own_prices.py.

The prices are simulated (landfall.prices.synthetic_prices), not market data; by default no
abnormal return is injected, so the run should find nothing. The file goes to the git-ignored
data/ directory. HCI starts in 2008 to exercise assets with a shorter history.

Usage: python scripts/make_synthetic_prices.py [--assets docs/own_prices_assets.csv]
           [--out data/synthetic/own_prices_synthetic.csv] [--seed 20261105] [--effect TRV=-0.05]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

from landfall import pipeline as pl
from landfall.prices import synthetic_prices
from landfall.provenance import REPO


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--assets", default=str(REPO / "docs" / "own_prices_assets.csv"))
    ap.add_argument("--out", default=str(REPO / "data" / "synthetic" /
                                         "own_prices_synthetic.csv"))
    ap.add_argument("--market", default="SPX")
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--effect", action="append", default=[],
                    help="ASSET=CAR, abnormal return spread over days 0..+10 of every event")
    args = ap.parse_args()
    names = [a.name for a in pl.load_assets(args.assets)]
    effects = {k: float(v) for k, v in (e.split("=") for e in args.effect)}
    events = pd.read_csv(REPO / "results" / "real_eventstudy" / "events.csv",
                         parse_dates=["time"])
    df = synthetic_prices(names, args.market, events["time"], args.seed, effects,
                          starts={"HCI": "2008-07-01"} if "HCI" in names else None)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, float_format="%.6f", date_format="%Y-%m-%d")
    print(f"wrote {out}: SYNTHETIC prices, {len(df)} dates, {df.shape[1]} series")
    return 0


if __name__ == "__main__":
    sys.exit(main())
