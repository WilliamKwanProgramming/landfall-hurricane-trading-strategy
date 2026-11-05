"""Point-in-time US landfall probabilities from the statistical forecast, for trading signals.

For every forecast sample (synoptic fix of a tropical or subtropical cyclone, see
landfall.forecast.build_samples) of a season s, the model fitted on seasons before s gives a
track forecast; `n_draws` joint error vectors drawn (with replacement, seeded by (seed, s)) from
that model's training residuals give sampled tracks; the probability is the fraction of sampled
tracks whose centre enters land inside the US coast polygon within 120 h
(landfall.exposure.us_landfall_flags, hourly linear interpolation). The forecast of fix t is
treated as public at t + `delay_hours` (default 3 h, the assumption used for NHC advisories).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from landfall import forecast as F
from landfall.exposure import us_landfall_flags
from landfall.geo import LandGrid, offset_latlon

NODE_H = np.array([0.0, *F.LEADS_H])


def _sampled_positions(samples: pd.DataFrame, preds: dict, residuals: np.ndarray,
                       draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """lat, lon of shape (n_samples * n_draws, 7) (sample-major)."""
    n, k = len(samples), len(draws)
    lat0 = samples["lat0"].to_numpy()
    lon0 = samples["lon0"].to_numpy()
    e = np.zeros((n, k, len(NODE_H)))
    nn = np.zeros((n, k, len(NODE_H)))
    r = residuals[draws]  # (k, leads, 3)
    for i, L in enumerate(F.LEADS_H):
        e[:, :, i + 1] = preds[(L, "east")][:, None] + r[None, :, i, 0]
        nn[:, :, i + 1] = preds[(L, "north")][:, None] + r[None, :, i, 1]
    lat, lon = offset_latlon(lat0[:, None, None], lon0[:, None, None], e, nn)
    return lat.reshape(n * k, -1), lon.reshape(n * k, -1)


def landfall_probabilities(samples: pd.DataFrame, seasons: range, grid: LandGrid,
                           n_draws: int = 200, seed: int = 0,
                           train_start: int = F.TRAIN_START, delay_hours: float = 3.0
                           ) -> pd.DataFrame:
    """US landfall probability within 120 h for every sample of the given seasons."""
    parts = []
    for s in seasons:
        te = samples[samples["season"] == s]
        if te.empty:
            continue
        model = F.fit_model(samples, s, train_start)
        preds = model.predict_targets(te)
        rng = np.random.default_rng([seed, s])
        draws = rng.integers(0, len(model.residuals), n_draws)
        lat, lon = _sampled_positions(te, preds, model.residuals, draws)
        flags = us_landfall_flags(NODE_H, lat, lon, grid).reshape(len(te), n_draws)
        out = te[["storm_id", "season", "time", "lat0", "lon0", "vmax0"]].copy()
        out["p_us_landfall"] = flags.mean(axis=1)
        out["available_time"] = out["time"] + pd.Timedelta(hours=delay_hours)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def first_triggers(probs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """For each storm, the first forecast whose probability is >= threshold (time order)."""
    hit = probs[probs["p_us_landfall"] >= threshold].sort_values("time")
    return hit.groupby("storm_id", as_index=False).first()

