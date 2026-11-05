#!/usr/bin/env python3
"""Experiment (b): skill of the point-in-time forecast model on real HURDAT2 tracks.

Rolling origin by season: every season 1990-2024 is forecast by a model fitted on 1950 up to the
previous season only. Mean great-circle track error (km) and mean absolute intensity error (kt)
by lead for the model, persistence and climatology, on the same (homogeneous) samples, with 95%
percentile intervals from a bootstrap that resamples whole storms (2000 replications). Real data
(best tracks); truth is the best track, not operational verification.

Usage: python experiments/forecast_skill.py [--seed 20261105] [--first 1990] [--last 2024]
"""

from __future__ import annotations

import argparse
import sys

from landfall import forecast as F
from landfall.data.tracks import load_combined_tracks
from landfall.provenance import results_dir, write_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--first", type=int, default=1990)
    ap.add_argument("--last", type=int, default=2024)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    tracks, _ = load_combined_tracks()
    samples = F.build_samples(tracks)
    fc = F.rolling_forecasts(samples, range(args.first, args.last + 1))
    err = F.forecast_errors(fc)
    out = results_dir("forecast_skill")
    track = F.storm_block_bootstrap_ci(err, "track_km", args.n_boot, args.seed)
    inten = F.storm_block_bootstrap_ci(err, "int_abs_kt", args.n_boot, args.seed + 1)
    track.to_csv(out / "track_error_km.csv", index=False, float_format="%.2f")
    inten.to_csv(out / "intensity_error_kt.csv", index=False, float_format="%.3f")
    by_season = err[err.lead_h.isin([24, 72])].groupby(["lead_h", "season", "method"])[
        "track_km"].mean().unstack("method")
    by_season.to_csv(out / "track_error_by_season_km.csv", float_format="%.2f")
    write_manifest(out, args.seed, {
        "seasons_evaluated": f"{args.first}-{args.last}",
        "training": f"{F.TRAIN_START} .. season - 1 (expanding)",
        "ridge_lambda": F.RIDGE_LAMBDA, "bootstrap": f"{args.n_boot} storm-block resamples",
        "samples": len(fc)})
    print(track.to_string())
    print(inten.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
