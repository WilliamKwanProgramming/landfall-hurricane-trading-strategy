"""Exposure scores: expected loss-relevant wind at an entity's points, given a forecast.

An entity is a set of points with weights (CSV with columns point, lat, lon, weight; comment
lines start with #). Two illustrative maps ship in landfall/exposure_maps/; their weights are
invented assumptions, not derived from any company's filings.

Score at time t, from the forecast available at t only:

    S_t = (1/N) sum_n sum_j w_j f(W_nj),

where W_nj is the maximum surface wind at point j for the n-th sampled storm (forecast plus a
joint error vector drawn from the training-season residuals, landfall.forecast.sample_tracks),
and f is the damage function of Emanuel (2011): f(V) = y^3 / (1 + y^3) with
y = max(V - 50 kt, 0) / (150 kt - 50 kt). S_t is a fraction-of-value-lost proxy in [0, 1]; its
Monte Carlo standard error is reported with it.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from landfall import footprint as fp
from landfall.data.atcf import TrackForecast
from landfall.events import in_polygon
from landfall.forecast import sample_tracks
from landfall.geo import LandGrid

MAPS_DIR = pathlib.Path(__file__).with_name("exposure_maps")
V_THRESH_KT = 50.0
V_HALF_KT = 150.0


def load_exposure_map(name_or_path: str | pathlib.Path) -> pd.DataFrame:
    """Read an exposure map by file path or by the name of a bundled map."""
    p = pathlib.Path(name_or_path)
    if not p.exists():
        p = MAPS_DIR / f"{name_or_path}.csv"
    df = pd.read_csv(p, comment="#")
    if list(df.columns) != ["point", "lat", "lon", "weight"]:
        raise ValueError(f"{p}: columns must be point,lat,lon,weight")
    if (df["weight"] < 0).any() or not np.isclose(df["weight"].sum(), 1.0):
        raise ValueError(f"{p}: weights must be non-negative and sum to 1")
    return df


def emanuel_damage(v_kt: ArrayLike, v_thresh: float = V_THRESH_KT,
                   v_half: float = V_HALF_KT) -> NDArray:
    """Emanuel (2011) damage fraction f = y^3 / (1 + y^3), y = max(V - Vt, 0) / (Vh - Vt)."""
    y = np.maximum(np.asarray(v_kt, dtype=float) - v_thresh, 0.0) / (v_half - v_thresh)
    return y ** 3 / (1.0 + y ** 3)


@dataclass(frozen=True)
class ExposureResult:
    score: float
    score_se: float
    n_samples: int
    p_ge_34: NDArray  # per point
    p_ge_64: NDArray  # per point
    p_us_landfall: float


def us_landfall_flags(node_h: NDArray, lat: NDArray, lon: NDArray, grid: LandGrid,
                      dt_h: float = 1.0) -> NDArray:
    """For each sampled track (rows), whether its centre enters land inside the US coast
    polygon within the horizon (a land step after a sea step, linear interpolation every dt_h)."""
    t, seg, frac = fp.substeps(node_h, dt_h)
    la = lat[:, seg] + (lat[:, seg + 1] - lat[:, seg]) * frac
    lo = lon[:, seg] + (lon[:, seg + 1] - lon[:, seg]) * frac
    land = grid.is_land(la, lo)
    enter = land[:, 1:] & ~land[:, :-1]
    us = in_polygon(la[:, 1:].ravel(), lo[:, 1:].ravel()).reshape(enter.shape)
    return (enter & us).any(axis=1)


def forecast_exposure(fc: TrackForecast, residuals: NDArray, draws: NDArray, rm_km: float,
                      b: float, emap: pd.DataFrame, grid: LandGrid, dt_h: float = 0.5,
                      engine: str = "numpy") -> ExposureResult:
    """Exposure score of `emap` from forecast `fc` using residual vectors residuals[draws]."""
    lat, lon, vmax = sample_tracks(fc, residuals, draws)
    n = len(draws)
    storms = fp.StormSet(np.asarray(fc.leads_h, dtype=float), lat, lon, vmax,
                         np.full(n, rm_km), np.full(n, b))
    lat_p, lon_p = emap["lat"].to_numpy(), emap["lon"].to_numpy()
    if engine == "cpp":
        mw = fp.max_wind_cpp(storms, lat_p, lon_p, grid, dt_h)
    else:
        mw = fp.max_wind_numpy(storms, lat_p, lon_p, grid, dt_h)
    per = emanuel_damage(mw) @ emap["weight"].to_numpy()
    ex = fp.exceedance(mw, [34.0, 64.0])
    p_us = float(us_landfall_flags(storms.node_h, lat, lon, grid).mean())
    return ExposureResult(float(per.mean()), float(per.std(ddof=1) / np.sqrt(n)) if n > 1 else
                          float("nan"), n, ex[0], ex[1], p_us)
