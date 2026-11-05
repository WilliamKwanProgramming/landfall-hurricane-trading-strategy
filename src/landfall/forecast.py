"""Point-in-time statistical track and intensity forecasts (CLIPER-like) and their evaluation.

Without archived operational forecasts, the forecast that was "available" at time t is produced
by a linear model fitted only on seasons before the season of t, from the storm state known at
t. Truth is the best-track position and intensity at t + lead; the model never sees it.

Samples. For every synoptic fix t of a storm whose status is tropical or subtropical (TD, TS, HU,
SD, SS) and that has synoptic fixes at t - 12 h and t - 24 h:

- features (`FEATURES`): lat, lon, vmax, the east/north displacement (km) over the previous 12 h
  and 24 h, the change in vmax over the previous 12 h and 24 h, sin and cos of the day of the
  year, and the quadratic terms lat^2, lon^2, lat*lon, vmax^2, lat*vmax;
- targets at each lead L in `LEADS_H` (12, 24, 48, 72, 96, 120 h) when the storm has a synoptic
  fix at t + L (any status): east and north displacement (km, on the plane tangent at the
  position at t, see landfall.geo.displacement_km) and the change in vmax (kt).

Model. For each lead and target separately, ridge regression on features standardised with the
training means and standard deviations, intercept not penalised:
beta = (Z'Z + lambda I)^{-1} Z'(y - ybar), lambda = `RIDGE_LAMBDA` (fixed in advance, not tuned).
Training seasons are `train_start` .. season(t) - 1 (expanding window).

Errors. The in-sample residual vectors (all leads jointly, from training samples that have every
lead) form the empirical error distribution; adding them to a forecast gives sampled tracks
(the "cone"). In-sample residuals understate out-of-sample errors slightly; with about ten
thousand training samples and 16 features the difference is small, but it is not zero.

Baselines. Persistence: the previous 12-h motion continued linearly, vmax unchanged.
Climatology: the mean training-season displacement and intensity change at that lead for samples
in the same 5 x 5 degree box (all training samples if the box has fewer than 10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from landfall.data.atcf import TrackForecast
from landfall.data.hurdat2 import TROPICAL_STATUS, synoptic
from landfall.geo import displacement_km, haversine_km, offset_latlon

LEADS_H = (12, 24, 48, 72, 96, 120)
FEATURES = ("lat", "lon", "vmax", "u12", "v12", "u24", "v24", "dv12", "dv24", "sin_doy",
            "cos_doy", "lat2", "lon2", "latlon", "vmax2", "latvmax")
TARGETS = ("east", "north", "dv")
RIDGE_LAMBDA = 1.0
TRAIN_START = 1950


def build_samples(tracks: pd.DataFrame) -> pd.DataFrame:
    """Features at each eligible fix and, where they exist, the targets at every lead.

    Uses synoptic fixes only. Columns: storm_id, season, time, lat0, lon0, vmax0, the features,
    and for each lead L the columns east_L, north_L, dv_L, lat_L, lon_L (NaN when the storm has
    no fix at t + L). Features use only fixes at t, t - 12 h and t - 24 h of the same storm.
    """
    s = synoptic(tracks).dropna(subset=["lat", "lon", "vmax_kt"])
    s = s.sort_values(["storm_id", "time"])
    keys = pd.MultiIndex.from_arrays([s["storm_id"], s["time"]])
    lat = pd.Series(s["lat"].to_numpy(), index=keys)
    lon = pd.Series(s["lon"].to_numpy(), index=keys)
    vmx = pd.Series(s["vmax_kt"].to_numpy(), index=keys)

    def at(series: pd.Series, hours: int) -> np.ndarray:
        k = pd.MultiIndex.from_arrays([s["storm_id"], s["time"] + pd.Timedelta(hours=hours)])
        return series.reindex(k).to_numpy()

    lat0, lon0, v0 = s["lat"].to_numpy(), s["lon"].to_numpy(), s["vmax_kt"].to_numpy()
    out = pd.DataFrame({"storm_id": s["storm_id"].to_numpy(), "season": s["season"].to_numpy(),
                        "time": s["time"].to_numpy(), "status": s["status"].to_numpy(),
                        "lat0": lat0, "lon0": lon0, "vmax0": v0})
    for h in (12, 24):
        e, n = displacement_km(at(lat, -h), at(lon, -h), lat0, lon0)
        out[f"u{h}"], out[f"v{h}"] = e, n
        out[f"dv{h}"] = v0 - at(vmx, -h)
    doy = pd.DatetimeIndex(out["time"]).dayofyear.to_numpy()
    out["sin_doy"], out["cos_doy"] = np.sin(2 * np.pi * doy / 365.25), np.cos(
        2 * np.pi * doy / 365.25)
    out["lat"], out["lon"], out["vmax"] = lat0, lon0, v0
    out["lat2"], out["lon2"], out["latlon"] = lat0 ** 2, lon0 ** 2, lat0 * lon0
    out["vmax2"], out["latvmax"] = v0 ** 2, lat0 * v0
    for L in LEADS_H:
        la, lo, vv = at(lat, L), at(lon, L), at(vmx, L)
        e, n = displacement_km(lat0, lon0, la, lo)
        out[f"east_{L}"], out[f"north_{L}"], out[f"dv_{L}"] = e, n, vv - v0
        out[f"lat_{L}"], out[f"lon_{L}"] = la, lo
    ok = out["status"].isin(TROPICAL_STATUS) & out[list(FEATURES)].notna().all(axis=1)
    return out[ok].reset_index(drop=True)


@dataclass
class RidgeFit:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray  # (n_features,) on standardised features
    intercept: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + ((x - self.mean) / self.scale) @ self.coef


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> RidgeFit:
    """Ridge regression with standardised features and an unpenalised intercept."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    ybar = float(y.mean())
    coef = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (y - ybar))
    return RidgeFit(mean, scale, coef, ybar)


@dataclass
class PointInTimeModel:
    """Models for every (lead, target) fitted on seasons `first_season` .. `last_season`."""

    first_season: int
    last_season: int
    fits: dict[tuple[int, str], RidgeFit]
    residuals: np.ndarray  # (n, len(LEADS_H), 3): joint in-sample errors (east, north, dv)
    clim: dict  # climatology tables

    def predict_targets(self, samples: pd.DataFrame) -> dict[tuple[int, str], np.ndarray]:
        x = samples[list(FEATURES)].to_numpy(dtype=float)
        return {k: f.predict(x) for k, f in self.fits.items()}


def fit_model(samples: pd.DataFrame, test_season: int, train_start: int = TRAIN_START,
              lam: float = RIDGE_LAMBDA) -> PointInTimeModel:
    """Fit on seasons train_start .. test_season - 1 (never the test season or later)."""
    tr = samples[(samples["season"] >= train_start) & (samples["season"] < test_season)]
    if tr.empty:
        raise ValueError(f"no training samples before {test_season}")
    x = tr[list(FEATURES)].to_numpy(dtype=float)
    fits = {}
    for L in LEADS_H:
        for tgt in TARGETS:
            y = tr[f"{tgt}_{L}"].to_numpy(dtype=float)
            ok = np.isfinite(y)
            fits[(L, tgt)] = fit_ridge(x[ok], y[ok], lam)
    cols = [f"{tgt}_{L}" for L in LEADS_H for tgt in TARGETS]
    full = tr[tr[cols].notna().all(axis=1)]
    xf = full[list(FEATURES)].to_numpy(dtype=float)
    res = np.empty((len(full), len(LEADS_H), 3))
    for i, L in enumerate(LEADS_H):
        for j, tgt in enumerate(TARGETS):
            res[:, i, j] = full[f"{tgt}_{L}"].to_numpy() - fits[(L, tgt)].predict(xf)
    return PointInTimeModel(int(tr["season"].min()), int(tr["season"].max()), fits, res,
                            _climatology(tr))


def _box(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return (np.floor(lat / 5.0).astype(int) * 1000 + np.floor(lon / 5.0).astype(int))


def _climatology(tr: pd.DataFrame) -> dict:
    box = _box(tr["lat0"].to_numpy(), tr["lon0"].to_numpy())
    table = {}
    for L in LEADS_H:
        for tgt in TARGETS:
            y = tr[f"{tgt}_{L}"]
            ok = y.notna().to_numpy()
            df = pd.DataFrame({"box": box[ok], "y": y.to_numpy()[ok]})
            g = df.groupby("box")["y"].agg(["mean", "size"])
            table[(L, tgt)] = (g["mean"][g["size"] >= 10].to_dict(), float(df["y"].mean()))
    return table


def climatology_targets(model: PointInTimeModel, samples: pd.DataFrame
                        ) -> dict[tuple[int, str], np.ndarray]:
    box = _box(samples["lat0"].to_numpy(), samples["lon0"].to_numpy())
    out = {}
    for k, (by_box, overall) in model.clim.items():
        out[k] = np.array([by_box.get(b, overall) for b in box])
    return out


def persistence_targets(samples: pd.DataFrame) -> dict[tuple[int, str], np.ndarray]:
    out = {}
    for L in LEADS_H:
        out[(L, "east")] = samples["u12"].to_numpy() * (L / 12.0)
        out[(L, "north")] = samples["v12"].to_numpy() * (L / 12.0)
        out[(L, "dv")] = np.zeros(len(samples))
    return out


def to_track_forecast(sample: pd.Series, targets: dict[tuple[int, str], float],
                      delay_hours: float = 0.0, source: str = "cliper-ridge") -> TrackForecast:
    """Convert predicted displacements of one sample into positions (a TrackForecast)."""
    e = np.array([0.0] + [targets[(L, "east")] for L in LEADS_H])
    n = np.array([0.0] + [targets[(L, "north")] for L in LEADS_H])
    dv = np.array([0.0] + [targets[(L, "dv")] for L in LEADS_H])
    lat, lon = offset_latlon(sample["lat0"], sample["lon0"], e, n)
    t = pd.Timestamp(sample["time"])
    return TrackForecast(sample["storm_id"], t, t + pd.Timedelta(hours=delay_hours),
                         np.array([0.0, *LEADS_H]), lat, lon,
                         np.maximum(sample["vmax0"] + dv, 0.0), source)


def rolling_forecasts(samples: pd.DataFrame, seasons: range, train_start: int = TRAIN_START,
                      lam: float = RIDGE_LAMBDA) -> pd.DataFrame:
    """Out-of-sample forecasts for every sample of each season in `seasons`.

    Each season's samples are forecast by a model fitted on earlier seasons only. Returns the
    test samples with columns pred_{model}_{target}_{L} for model in (ridge, pers, clim).
    """
    parts = []
    for s in seasons:
        te = samples[samples["season"] == s]
        if te.empty:
            continue
        m = fit_model(samples, s, train_start, lam)
        te = te.copy()
        for name, pred in (("ridge", m.predict_targets(te)), ("pers", persistence_targets(te)),
                           ("clim", climatology_targets(m, te))):
            for (L, tgt), v in pred.items():
                te[f"pred_{name}_{tgt}_{L}"] = v
        parts.append(te)
    return pd.concat(parts, ignore_index=True)


def forecast_errors(fc: pd.DataFrame, methods: tuple[str, ...] = ("ridge", "pers", "clim")
                    ) -> pd.DataFrame:
    """Long table of track error (km, great circle) and absolute intensity error (kt).

    Only samples verified at that lead (truth exists) are used; every method is scored on the
    same samples (homogeneous comparison).
    """
    rows = []
    for L in LEADS_H:
        ok = fc[f"lat_{L}"].notna() & fc[f"dv_{L}"].notna()
        d = fc[ok]
        for m in methods:
            la, lo = offset_latlon(d["lat0"], d["lon0"], d[f"pred_{m}_east_{L}"],
                                   d[f"pred_{m}_north_{L}"])
            rows.append(pd.DataFrame({
                "storm_id": d["storm_id"].to_numpy(), "season": d["season"].to_numpy(),
                "time": d["time"].to_numpy(), "lead_h": L, "method": m,
                "track_km": haversine_km(la, lo, d[f"lat_{L}"], d[f"lon_{L}"]),
                "int_abs_kt": np.abs(d[f"pred_{m}_dv_{L}"] - d[f"dv_{L}"]).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def storm_block_bootstrap_ci(err: pd.DataFrame, value: str, n_boot: int = 2000,
                             seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """Mean error by lead and method with percentile CIs from resampling whole storms.

    Errors of forecasts from the same storm are correlated, so the resampling unit is the storm
    (all its samples at that lead). Also returns the paired difference ridge - pers with its CI
    (same storm draws for both methods).
    """
    rng = np.random.default_rng(seed)
    a = (1 - level) / 2
    rows = []
    for L, d in err.groupby("lead_h"):
        storms = d["storm_id"].unique()
        sid_index = {s: i for i, s in enumerate(storms)}
        per = {}
        for m, dm in d.groupby("method"):
            idx = dm["storm_id"].map(sid_index).to_numpy()
            per[m] = (np.bincount(idx, weights=dm[value].to_numpy(), minlength=len(storms)),
                      np.bincount(idx, minlength=len(storms)))
        draws = rng.integers(0, len(storms), size=(n_boot, len(storms)))
        counts = np.stack([np.bincount(r, minlength=len(storms)) for r in draws])
        boot = {m: (counts @ s) / (counts @ c) for m, (s, c) in per.items()}
        for m, (s, c) in per.items():
            lo, hi = np.quantile(boot[m], [a, 1 - a])
            rows.append({"lead_h": L, "method": m, "n": int(c.sum()), "storms": len(storms),
                         "mean": s.sum() / c.sum(), "ci_lo": lo, "ci_hi": hi})
        if "ridge" in per and "pers" in per:
            diff = boot["ridge"] - boot["pers"]
            point = per["ridge"][0].sum() / per["ridge"][1].sum() - (
                per["pers"][0].sum() / per["pers"][1].sum())
            lo, hi = np.quantile(diff, [a, 1 - a])
            rows.append({"lead_h": L, "method": "ridge-pers", "n": int(per["ridge"][1].sum()),
                         "storms": len(storms), "mean": point, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def sample_tracks(fc: TrackForecast, residuals: np.ndarray, draws: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sampled tracks: forecast plus joint residual vectors `residuals[draws]`.

    Returns lat, lon, vmax arrays of shape (len(draws), 1 + len(LEADS_H)); column 0 is the
    initial state (no error).
    """
    e0, n0 = displacement_km(fc.lat[0], fc.lon[0], fc.lat[1:], fc.lon[1:])
    r = residuals[draws]  # (N, leads, 3)
    e = np.concatenate([np.zeros((len(draws), 1)), e0[None, :] + r[:, :, 0]], axis=1)
    n = np.concatenate([np.zeros((len(draws), 1)), n0[None, :] + r[:, :, 1]], axis=1)
    v = np.concatenate([np.full((len(draws), 1), fc.vmax_kt[0]),
                        np.maximum(fc.vmax_kt[None, 1:] + r[:, :, 2], 0.0)], axis=1)
    lat, lon = offset_latlon(fc.lat[0], fc.lon[0], e, n)
    return lat, lon, v
