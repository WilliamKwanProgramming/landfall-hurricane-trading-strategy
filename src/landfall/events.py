"""Landfall definitions: HURDAT2 landfall records and a geometric sea-to-land crossing rule.

Two definitions, compared in experiments/landfall_counts.py:

1. Flagged: entries with record identifier "L" ("center of system crossing a coastline",
   NHC HURDAT2 format document). Available in the rda mirror (1851-2019) and the NHC file.
2. Geometric: the track through the synoptic (00/06/12/18 UTC) fixes of a storm is linearly
   interpolated in latitude, longitude, time and intensity at `step_minutes` steps. A landfall is
   a step whose position is on land (land mask from the `global-land-mask` package, see
   landfall.geo) after at least `min_sea_hours` of consecutive steps at sea. The first fix of a
   storm over land does not count. Both definitions are applied to every status (the "L" records
   include tropical storms, depressions and a few extratropical systems); filters by status and
   intensity are applied afterwards.

"US coast" region: a landfall point inside `US_COAST_POLYGON`, a hand-drawn polygon (vertices
below, lon/lat degrees) that encloses the coast of the contiguous United States from the Rio
Grande to the Maine-New Brunswick border, passing offshore between Florida and the Bahamas and
south of the Florida Keys and Dry Tortugas, and inland far behind the coast. It is meant for
points on the coastline only; its inland edges are not the national border.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from landfall.data.hurdat2 import synoptic
from landfall.geo import LandGrid, haversine_km

US_COAST_POLYGON: tuple[tuple[float, float], ...] = (
    (-97.14, 25.96),  # Rio Grande mouth
    (-97.50, 25.85),  # up the Rio Grande (Brownsville)
    (-99.50, 27.50),  # Laredo
    (-101.00, 29.80),  # Del Rio
    (-95.00, 35.00),  # inland
    (-85.00, 37.00),  # inland
    (-78.00, 42.00),  # inland
    (-71.50, 45.00),  # Canada border (approximate)
    (-67.80, 45.50),  # Canada border (approximate)
    (-67.00, 44.75),  # Passamaquoddy Bay
    (-66.50, 44.00),  # offshore, Gulf of Maine (Nova Scotia stays outside)
    (-69.00, 41.00),  # offshore, east of Cape Cod and Nantucket
    (-74.00, 38.50),  # offshore
    (-75.00, 35.00),  # offshore, east of Cape Hatteras
    (-79.60, 31.00),  # offshore
    (-79.60, 24.30),  # between Florida and the Bahamas (Bimini at 79.3W stays outside)
    (-83.20, 24.30),  # south of the Florida Keys and the Dry Tortugas
    (-96.50, 25.50),  # Gulf of Mexico, offshore of the Mexican coast
)


def in_polygon(lat: np.ndarray, lon: np.ndarray,
               poly: tuple[tuple[float, float], ...] = US_COAST_POLYGON) -> np.ndarray:
    """Even-odd ray-casting point-in-polygon test (polygon vertices as (lon, lat))."""
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    inside = np.zeros(lat.shape, dtype=bool)
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        crosses = (y1 > lat) != (y2 > lat)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (lon < x_at)
    return inside


EVENT_COLUMNS = ["storm_id", "name", "season", "time", "lat", "lon", "vmax_kt", "status",
                 "definition"]


def flagged_landfalls(tracks: pd.DataFrame) -> pd.DataFrame:
    """Landfalls from HURDAT2 record identifier "L" (definition 1)."""
    if tracks["record"].isna().all():
        raise ValueError("these tracks carry no record identifiers (dplyr mirror?)")
    lf = tracks[tracks["record"] == "L"].copy()
    lf["definition"] = "flag"
    return lf[EVENT_COLUMNS].reset_index(drop=True)


def interpolate_storm(storm: pd.DataFrame, step_minutes: int = 10) -> pd.DataFrame:
    """Linear interpolation of one storm's synoptic fixes at a fixed time step.

    Latitude, longitude and maximum wind are interpolated linearly in time; status is carried
    forward from the earlier fix. Fixes with missing position are skipped; missing wind is
    interpolated from the neighbouring fixes that have it.
    """
    s = synoptic(storm).dropna(subset=["lat", "lon"]).sort_values("time")
    if len(s) == 0:
        return s.iloc[:0][["time", "lat", "lon", "vmax_kt", "status"]]
    t = s["time"].to_numpy().astype("datetime64[s]").astype(np.int64).astype(float)
    step = 60.0 * step_minutes
    grid = np.arange(t[0], t[-1] + 0.5 * step, step)
    v = s["vmax_kt"].to_numpy(dtype=float)
    ok = np.isfinite(v)
    vmax = np.interp(grid, t[ok], v[ok]) if ok.any() else np.full(grid.shape, np.nan)
    idx = np.searchsorted(t, grid, side="right") - 1
    return pd.DataFrame({
        "time": grid.astype(np.int64).astype("datetime64[s]"),
        "lat": np.interp(grid, t, s["lat"].to_numpy(dtype=float)),
        "lon": np.interp(grid, t, s["lon"].to_numpy(dtype=float)),
        "vmax_kt": vmax,
        "status": s["status"].to_numpy()[np.clip(idx, 0, len(s) - 1)],
    })


def geometric_landfalls(tracks: pd.DataFrame, grid: LandGrid, step_minutes: int = 10,
                        min_sea_hours: float = 1.0) -> pd.DataFrame:
    """Sea-to-land crossings of the interpolated synoptic track (definition 2)."""
    need = max(1, int(round(min_sea_hours * 60 / step_minutes)))
    out = []
    for sid, storm in tracks.groupby("storm_id", sort=False):
        p = interpolate_storm(storm, step_minutes)
        if len(p) < need + 1:
            continue
        land = grid.is_land(p["lat"].to_numpy(), p["lon"].to_numpy())
        sea_run = np.zeros(len(land), dtype=int)
        run = 0
        for k, is_l in enumerate(land):
            sea_run[k] = run
            run = 0 if is_l else run + 1
        hits = np.nonzero(land & (sea_run >= need))[0]
        for k in hits:
            out.append((sid, storm["name"].iloc[0], int(storm["season"].iloc[0]),
                        p["time"].iloc[k], p["lat"].iloc[k], p["lon"].iloc[k],
                        p["vmax_kt"].iloc[k], p["status"].iloc[k], "geometric"))
    return pd.DataFrame(out, columns=EVENT_COLUMNS)


def us_landfalls(events: pd.DataFrame, min_vmax_kt: float | None = None) -> pd.DataFrame:
    """Landfalls inside the US coast polygon, optionally at or above an intensity."""
    keep = in_polygon(events["lat"].to_numpy(), events["lon"].to_numpy())
    if min_vmax_kt is not None:
        keep &= events["vmax_kt"].to_numpy(dtype=float) >= min_vmax_kt
    return events[keep].reset_index(drop=True)


def match_definitions(flag: pd.DataFrame, geo: pd.DataFrame, max_hours: float = 6.0,
                      max_km: float = 150.0) -> pd.DataFrame:
    """Pair each flagged landfall with the nearest-in-time geometric one of the same storm.

    A pair counts when the times differ by at most `max_hours` and the positions by at most
    `max_km`; each geometric landfall is used at most once (greedy, closest in time first).
    Returns one row per flagged landfall with the matched geometric time and distance (NaT/NaN
    when unmatched).
    """
    rows = []
    used: set[int] = set()
    cand = []
    for i, f in flag.iterrows():
        g = geo[geo["storm_id"] == f["storm_id"]]
        for j, r in g.iterrows():
            dh = abs((r["time"] - f["time"]).total_seconds()) / 3600.0
            dk = float(haversine_km(f["lat"], f["lon"], r["lat"], r["lon"]))
            if dh <= max_hours and dk <= max_km:
                cand.append((dh, dk, i, j))
    cand.sort()
    pair: dict[int, tuple[int, float, float]] = {}
    for dh, dk, i, j in cand:
        if i in pair or j in used:
            continue
        pair[i] = (j, dh, dk)
        used.add(j)
    for i, f in flag.iterrows():
        j, dh, dk = pair.get(i, (None, np.nan, np.nan))
        rows.append({"storm_id": f["storm_id"], "flag_time": f["time"],
                     "geo_time": geo.loc[j, "time"] if j is not None else pd.NaT,
                     "hours_apart": dh, "km_apart": dk, "geo_index": j})
    return pd.DataFrame(rows)


def storms_with_us_hurricane_landfall(events: pd.DataFrame, min_vmax_kt: float = 64.0
                                      ) -> pd.DataFrame:
    """One row per storm with at least one US landfall at hurricane intensity (the first)."""
    us = us_landfalls(events, min_vmax_kt=min_vmax_kt)
    return us.sort_values("time").groupby("storm_id", as_index=False).first()


def dispersion_test(counts: np.ndarray) -> dict:
    """Poisson dispersion index test for counts per season.

    D = sum (x_i - xbar)^2 / xbar, which is approximately chi-square with n - 1 degrees of
    freedom when the counts are independent Poisson (Fisher's index of dispersion). Returns the
    mean, variance, variance/mean ratio, D, degrees of freedom and two-sided p-value.
    """
    from scipy import stats

    x = np.asarray(counts, dtype=float)
    n = x.size
    m = x.mean()
    d = float(((x - m) ** 2).sum() / m)
    df = n - 1
    p_upper = float(stats.chi2.sf(d, df))
    p_lower = float(stats.chi2.cdf(d, df))
    return {"n_seasons": n, "mean": m, "variance": float(x.var(ddof=1)),
            "ratio": float(x.var(ddof=1) / m), "D": d, "df": df,
            "p_two_sided": min(1.0, 2 * min(p_upper, p_lower))}
