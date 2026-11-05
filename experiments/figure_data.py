#!/usr/bin/env python3
"""Data for the README figures that no other experiment writes (scripts/make_figures.py plots).

1. Mean CAR by event day, days -5 .. t for t = -5 .. +10, of the pooled signed storm-relevant
   stocks and the pooled controls of the wave-1 study (docs/protocol.md, primary events; real
   data), with 95% stationary-bootstrap intervals (2,000 resamples, mean block 4 asset-events in
   event order, as the pooled test). Descriptive: the protocol's tests are in
   results/real_eventstudy/.
2. The 1,000 placebo values of the pooled signed mean CAR[0, +10]: the same draws as
   results/real_eventstudy/placebo.csv (same seed and code), which reports only their summary.
3. One footprint example: the statistical forecast of Hurricane Katrina (AL122005) issued at
   2005-08-27 12 UTC (the model fitted on 1950-2004), 200 sampled tracks, the C++ Monte Carlo
   kernel on a 0.2-degree grid: maximum wind of the central forecast track and the probability
   of at least 64 kt. Rm and B from the best-track state at the forecast time.

Usage: python experiments/figure_data.py [--seed 20261105]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from landfall import footprint as fp
from landfall import forecast as F
from landfall import pipeline as pl
from landfall.data.prices import load_skfolio_prices, simple_returns
from landfall.data.tracks import load_combined_tracks
from landfall.geo import atlantic_land_grid
from landfall.provenance import REPO, results_dir, write_manifest
from landfall.study import bootstrap as bs

STORM = {"HD": 1, "XOM": -1, "CVX": -1, "RRC": 1}
CONTROLS = ["MSFT", "JNJ", "MRK", "PFE"]
KATRINA, ISSUE = "AL122005", pd.Timestamp("2005-08-27 12:00")


def car_paths(datas, prim, seed):
    fits, _ = pl.fit_assets(datas, prim["time"])
    rows = []
    for group, signed in (("storm (signed)", True), ("control", False)):
        parts = [(d.asset, fits[d.asset.name]) for d in datas
                 if (d.asset.directional if signed else d.asset.group == "control")]
        ar = np.vstack([(a.sign if signed else 1) * f.mm.ar for a, f in parts])
        ev = np.concatenate([f.events for _, f in parts])
        ar = ar[np.argsort(ev, kind="stable")]
        cum = ar.cumsum(axis=1)
        for j, day in enumerate(range(-5, 11)):
            lo, hi = bs.bootstrap_mean_ci(cum[:, j], 4.0, 2000, seed)
            rows.append({"group": group, "day": day, "n": len(cum),
                         "mean_car": cum[:, j].mean(), "boot_lo": lo, "boot_hi": hi})
    return pd.DataFrame(rows)


def footprint_example(seed):
    tracks, _ = load_combined_tracks()
    samples = F.build_samples(tracks)
    s = samples[(samples.storm_id == KATRINA) & (samples.time == ISSUE)].iloc[0]
    model = F.fit_model(samples, 2005)
    targets = {k: float(v[0]) for k, v in model.predict_targets(s.to_frame().T).items()}
    fc = F.to_track_forecast(s, targets)
    best = tracks[(tracks.storm_id == KATRINA)]
    row = best[best.time == ISSUE].iloc[0]
    r34 = row[["r34_ne", "r34_se", "r34_sw", "r34_nw"]].to_numpy(dtype=float)
    rm, b = fp.storm_structure(float(row.vmax_kt), float(row.mslp_mb), float(row.lat), r34)
    rng = np.random.default_rng([seed, 2005])
    draws = rng.integers(0, len(model.residuals), 200)
    lat, lon, vmax = F.sample_tracks(fc, model.residuals, draws)
    glat, glon = np.meshgrid(np.arange(22.0, 36.01, 0.2), np.arange(-98.0, -80.99, 0.2),
                             indexing="ij")
    lat_p, lon_p = glat.ravel(), glon.ravel()
    grid = atlantic_land_grid(coarsen=6)
    node = np.asarray(fc.leads_h, dtype=float)
    mc = fp.max_wind_cpp(fp.StormSet(node, lat, lon, vmax, np.full(200, rm), np.full(200, b)),
                         lat_p, lon_p, grid)
    central = fp.max_wind_cpp(fp.StormSet(node, fc.lat[None], fc.lon[None], fc.vmax_kt[None],
                                          np.array([rm]), np.array([b])), lat_p, lon_p, grid)
    field = pd.DataFrame({"lat": lat_p, "lon": lon_p, "central_track_max_wind_kt": central[0],
                          "p_ge_34kt": fp.exceedance(mc, [34.0])[0],
                          "p_ge_64kt": fp.exceedance(mc, [64.0])[0]})
    trk = pd.concat([
        pd.DataFrame({"kind": "forecast (central)", "lead_h": node, "lat": fc.lat,
                      "lon": fc.lon, "vmax_kt": fc.vmax_kt}),
        pd.DataFrame({"kind": "best track", "lead_h": (best.time - ISSUE).dt.total_seconds()
                      / 3600, "lat": best.lat, "lon": best.lon, "vmax_kt": best.vmax_kt})])
    meta = {"footprint_storm": f"{KATRINA} KATRINA, statistical forecast issued {ISSUE} UTC",
            "footprint_rm_km": f"{rm:.1f}", "footprint_holland_b": f"{b:.3f}",
            "footprint_draws": 200, "footprint_grid": "0.2 degree, 22-36N, 98-81W"}
    return field, trk, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261105)
    args = ap.parse_args()
    out = results_dir("figure_data")

    prices, index = load_skfolio_prices()
    ret = simple_returns(prices)
    mkt = simple_returns(index).to_numpy()
    assets = [pl.Asset(a, "storm", s) for a, s in STORM.items()]
    assets += [pl.Asset(a, "control", 0) for a in CONTROLS]
    datas = [pl.AssetData(a, ret.index, ret[a.name].to_numpy(), mkt) for a in assets]
    events = pd.read_csv(REPO / "results" / "real_eventstudy" / "events.csv",
                         parse_dates=["time"])
    prim = events[events["kept_primary"]].sort_values("time").reset_index(drop=True)
    car_paths(datas, prim, args.seed).to_csv(out / "car_paths.csv", index=False,
                                              float_format="%.6g")

    excl = pd.read_csv(REPO / "results" / "landfall_counts" /
                       "us_landfalls_any_intensity_1985_2024.csv", parse_dates=["time"])
    _, signed = pl.placebo(datas, ret.index, prim["time"], excl["time"], (1991, 2022), 1000,
                           args.seed)
    pd.DataFrame({"placebo_set": np.arange(len(signed)), "pooled_signed_mean_car": signed}) \
        .to_csv(out / "placebo_pooled_signed.csv", index=False, float_format="%.6g")

    field, trk, meta = footprint_example(args.seed)
    field.to_csv(out / "footprint_katrina_field.csv", index=False, float_format="%.4f")
    trk.to_csv(out / "footprint_katrina_tracks.csv", index=False, float_format="%.4f")
    write_manifest(out, args.seed, {
        "data": "REAL: skfolio prices 1990-2022 and HURDAT2 tracks; footprint is model output",
        **meta})
    return 0


if __name__ == "__main__":
    sys.exit(main())
