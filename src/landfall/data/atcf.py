"""ATCF a-deck reader: official NHC forecasts (technique OFCL) with the time they became public.

Field order of the ATCF "Best Track/Objective Aid/Wind Radii Format" (NRL ATCF documentation,
science.nrlmry.navy.mil/atcf/docs/database/new/abrdeck.html, read on 2026-11-05), comma
separated, blanks allowed after the first eight fields:

 1 BASIN, 2 CY (cyclone number), 3 YYYYMMDDHH (warning date-time), 4 TECHNUM/MIN, 5 TECH,
 6 TAU (forecast hour), 7 LatN/S and 8 LonE/W (tenths of a degree with a hemisphere letter),
 9 VMAX (kt), 10 MSLP (mb), 11 TY (development level), 12 RAD (34, 50 or 64 kt),
 13 WINDCODE (NEQ = quadrants), 14-17 RAD1-RAD4 (NE, SE, SW, NW radii, n mi),
 18 RADP, 19 RRP, 20 MRD (radius of maximum wind, n mi), 21 GUSTS, 22 EYE, 23 SUBREGION,
 24 MAXSEAS, 25 INITIALS, 26 DIR, 27 SPEED, 28 STORMNAME, 29 DEPTH, 30-35 seas.

One forecast point can span several lines (one per wind-radius threshold); they are merged.
No a-deck file was available where this package was built: the reader is tested on a fixture
written from the documentation, not on a real file.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

ADECK_COLUMNS = ["storm_id", "basin", "cy", "issue_time", "tech", "tau_h", "lat", "lon",
                 "vmax_kt", "mslp_mb", "r34_ne", "r34_se", "r34_sw", "r34_nw", "r50_ne",
                 "r50_se", "r50_sw", "r50_nw", "r64_ne", "r64_se", "r64_sw", "r64_nw", "rmw_nm"]

DEFAULT_DELAY_HOURS = 3.0
"""Assumed time between the synoptic time of an NHC forecast and its public release. The
advisory for a 12 UTC forecast is normally issued at 15 UTC; this is an assumption, not a
measured value, and it is a parameter everywhere it is used."""


def _tenths(token: str, pos: str, neg: str) -> float:
    token = token.strip()
    if not token:
        return np.nan
    v = int(token[:-1]) / 10.0
    if token[-1] == neg:
        return -v
    if token[-1] != pos:
        raise ValueError(f"bad coordinate {token!r}")
    return v


def _int_or_nan(token: str, zero_missing: bool = False) -> float:
    token = token.strip()
    if not token:
        return np.nan
    v = float(int(token))
    return np.nan if zero_missing and v == 0 else v


def parse_adeck(text: str) -> pd.DataFrame:
    """Parse a-deck text into one row per (storm, issue time, technique, tau).

    MSLP 0 and MRD 0 are treated as missing. Wind radii come from lines with WINDCODE NEQ;
    full-circle (AAA) radii are deprecated in the format and are put in all four quadrants.
    """
    recs: dict[tuple, dict] = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        f = [t.strip() for t in line.split(",")]
        if len(f) < 8:
            raise ValueError(f"line {n}: {len(f)} fields, at least 8 required")
        basin, cy, dtg, tech, tau = f[0], int(f[1]), f[2], f[4], int(f[5])
        if len(dtg) != 10:
            raise ValueError(f"line {n}: bad date-time group {dtg!r}")
        issue = pd.Timestamp(f"{dtg[:4]}-{dtg[4:6]}-{dtg[6:8]} {dtg[8:10]}:00")
        key = (basin, cy, issue, tech, tau)
        rec = recs.get(key)
        if rec is None:
            rec = {c: np.nan for c in ADECK_COLUMNS}
            rec.update(storm_id=f"{basin}{cy:02d}{issue.year}", basin=basin, cy=cy,
                       issue_time=issue, tech=tech, tau_h=tau,
                       lat=_tenths(f[6], "N", "S"), lon=_tenths(f[7], "E", "W"))
            recs[key] = rec

        def get(i: int, f: list[str] = f) -> str:
            return f[i] if len(f) > i else ""

        if np.isnan(rec["vmax_kt"]):
            rec["vmax_kt"] = _int_or_nan(get(8))
        if np.isnan(rec["mslp_mb"]):
            rec["mslp_mb"] = _int_or_nan(get(9), zero_missing=True)
        if np.isnan(rec["rmw_nm"]):
            rec["rmw_nm"] = _int_or_nan(get(19), zero_missing=True)
        rad = get(11)
        if rad in ("34", "50", "64"):
            code = get(12).upper()
            radii = [_int_or_nan(get(13 + k)) for k in range(4)]
            if code == "AAA":
                radii = [radii[0]] * 4
            for q, v in zip(("ne", "se", "sw", "nw"), radii, strict=True):
                rec[f"r{rad}_{q}"] = v
    df = pd.DataFrame(list(recs.values()), columns=ADECK_COLUMNS)
    df["issue_time"] = pd.to_datetime(df["issue_time"]).astype("datetime64[s]")
    return df.sort_values(["storm_id", "issue_time", "tech", "tau_h"]).reset_index(drop=True)


def read_adeck(path: str | pathlib.Path) -> pd.DataFrame:
    """Read an a-deck file (plain text or .gz)."""
    path = pathlib.Path(path)
    if path.suffix == ".gz":
        import gzip

        return parse_adeck(gzip.decompress(path.read_bytes()).decode("ascii", "replace"))
    return parse_adeck(path.read_text(errors="replace"))


def official(adeck: pd.DataFrame) -> pd.DataFrame:
    """Rows of the official NHC forecast (technique OFCL)."""
    return adeck[adeck["tech"] == "OFCL"].reset_index(drop=True)


@dataclass(frozen=True)
class TrackForecast:
    """A track and intensity forecast as it was available at `available_time`.

    leads_h[0] == 0 is the initial state. Positions in degrees (lon east), vmax in kt. The
    statistical model in landfall.forecast and the ATCF reader both produce this type.
    """

    storm_id: str
    issue_time: pd.Timestamp
    available_time: pd.Timestamp
    leads_h: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    vmax_kt: np.ndarray
    source: str


def available_forecast(adeck: pd.DataFrame, storm_id: str, t: pd.Timestamp,
                       delay_hours: float = DEFAULT_DELAY_HOURS,
                       tech: str = "OFCL") -> TrackForecast | None:
    """The latest forecast of `tech` for `storm_id` that was public at time t.

    A forecast with synoptic time s counts as public from s + delay_hours. Returns None when no
    forecast was public yet.
    """
    rows = adeck[(adeck["storm_id"] == storm_id) & (adeck["tech"] == tech)]
    avail = rows["issue_time"] + pd.Timedelta(hours=delay_hours)
    rows = rows[avail <= pd.Timestamp(t)]
    if rows.empty:
        return None
    issue = rows["issue_time"].max()
    r = rows[rows["issue_time"] == issue].sort_values("tau_h")
    r = r[r["tau_h"] >= 0]
    return TrackForecast(storm_id, pd.Timestamp(issue),
                         pd.Timestamp(issue) + pd.Timedelta(hours=delay_hours),
                         r["tau_h"].to_numpy(dtype=float), r["lat"].to_numpy(dtype=float),
                         r["lon"].to_numpy(dtype=float), r["vmax_kt"].to_numpy(dtype=float),
                         source=f"atcf-{tech}")
