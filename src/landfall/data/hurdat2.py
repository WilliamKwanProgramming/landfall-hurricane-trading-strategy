"""HURDAT2 best-track data: parser for the NHC text format and loaders for two GitHub mirrors.

All loaders return the same schema (`TRACK_COLUMNS`), one row per best-track entry, sorted by
storm and time:

- storm_id: ATCF identifier such as "AL092011" (basin, number, year). The dplyr mirror has no
  identifier; see `load_dplyr_mirror`.
- name, season (year of the identifier), time (UTC, naive datetime64), record (one-letter record
  identifier: "" when blank, missing when the source does not carry it), status (two-letter code).
- lat (degrees north), lon (degrees east, so the western hemisphere is negative).
- vmax_kt (1-minute maximum sustained surface wind), mslp_mb (minimum central pressure).
- r34_ne ... r64_nw: wind radii (n mi) by quadrant; 0 means no wind of that strength in the
  quadrant, missing means not analysed.
- rmw_nm: radius of maximum wind (n mi), only in files that carry the 21st field.
- source: which loader produced the row.

Format (NHC, "Hurricane Database (HURDAT2) format", 2013 edition and later): a header line
`AL092011,              IRENE,     39,` (basin + ATCF number + year, name, number of entries)
followed by that many comma-separated data lines: date YYYYMMDD, time HHMM, record identifier,
status, latitude with N/S, longitude with E/W, maximum wind (kt), minimum pressure (mb), then the
34, 50 and 64 kt wind radii for the NE, SE, SW and NW quadrants. Since the 2022 release a 21st
field, radius of maximum wind (n mi), follows. Missing values are -999 (and -99 for the
unassigned intensities of some 1967 depressions).
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pandas as pd

RADII = [f"r{k}_{q}" for k in (34, 50, 64) for q in ("ne", "se", "sw", "nw")]
TRACK_COLUMNS = ["storm_id", "name", "season", "time", "record", "status", "lat", "lon",
                 "vmax_kt", "mslp_mb", *RADII, "rmw_nm", "source"]

RECORD_IDS = {"L": "landfall", "W": "maximum sustained wind", "P": "minimum central pressure",
              "I": "intensity peak in pressure and wind", "C": "closest approach to a coast",
              "S": "change of status", "G": "genesis", "T": "additional track detail"}
STATUS_CODES = ("TD", "TS", "HU", "EX", "SD", "SS", "LO", "WV", "DB")
TROPICAL_STATUS = ("TD", "TS", "HU", "SD", "SS")
"""Statuses of a tropical or subtropical cyclone (used to select fixes for the forecast model)."""

_MISSING = (-999.0, -99.0)


def _num(token: str) -> float:
    token = token.strip()
    if token == "":
        return np.nan
    v = float(token)
    return np.nan if v in _MISSING else v


def _lat(token: str) -> float:
    token = token.strip()
    v = float(token[:-1])
    if token[-1] == "S":
        return -v
    if token[-1] != "N":
        raise ValueError(f"bad latitude {token!r}")
    return v


def _lon(token: str) -> float:
    token = token.strip()
    v = float(token[:-1])
    if token[-1] == "W":
        return -v
    if token[-1] != "E":
        raise ValueError(f"bad longitude {token!r}")
    return v


_HEADER = re.compile(r"^\s*([A-Z]{2}\d{6})\s*,\s*([^,]*?)\s*,\s*(\d+)\s*,?\s*$")


def parse_hurdat2(text: str, source: str = "hurdat2-text") -> pd.DataFrame:
    """Parse HURDAT2 text into the common schema. Raises ValueError on a malformed file.

    Checks that every storm has exactly the announced number of data lines.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    rows: list[list] = []
    i = 0
    while i < len(lines):
        m = _HEADER.match(lines[i])
        if not m:
            raise ValueError(f"line {i + 1}: expected a header line, got {lines[i]!r}")
        storm_id, name, n = m.group(1), m.group(2).strip(), int(m.group(3))
        if i + n >= len(lines):
            raise ValueError(f"{storm_id}: header announces {n} lines, file ends early")
        for k in range(1, n + 1):
            if _HEADER.match(lines[i + k]):
                raise ValueError(f"{storm_id}: header announces {n} lines, found {k - 1}")
            f = [t.strip() for t in lines[i + k].split(",")]
            while f and f[-1] == "":
                f.pop()
            if len(f) not in (20, 21):
                raise ValueError(f"{storm_id}: data line {k} has {len(f)} fields, expected 20/21")
            date, hhmm = f[0], f[1].zfill(4)
            time = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]} {hhmm[:2]}:{hhmm[2:]}")
            status = f[3]
            if status not in STATUS_CODES:
                raise ValueError(f"{storm_id}: unknown status {status!r}")
            radii = [_num(t) for t in f[8:20]]
            rmw = _num(f[20]) if len(f) == 21 else np.nan
            rows.append([storm_id, name, int(storm_id[4:]), time, f[2], status, _lat(f[4]),
                         _lon(f[5]), _num(f[6]), _num(f[7]), *radii, rmw, source])
        i += n + 1
    return _finish(pd.DataFrame(rows, columns=TRACK_COLUMNS))


def read_hurdat2(path: str | pathlib.Path) -> pd.DataFrame:
    """Read an official HURDAT2 text file (for example the file fetched with --source nhc)."""
    return parse_hurdat2(pathlib.Path(path).read_text(), source="nhc")


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"]).astype("datetime64[s]")
    for c in ["lat", "lon", "vmax_kt", "mslp_mb", *RADII, "rmw_nm"]:
        df[c] = df[c].astype(float)
    df["season"] = df["season"].astype(int)
    df = df.sort_values(["season", "storm_id", "time"], kind="stable").reset_index(drop=True)
    return df[TRACK_COLUMNS]


def load_rda_mirror(path: str | pathlib.Path) -> tuple[pd.DataFrame, list[str]]:
    """Load the R data file `AL.rda` of github.com/timtrice/HURDAT (HURDAT2 1851-2019).

    Columns there: Key, Name, DateTime, Record, Status, Lat, Lon, Wind, Pressure, NE34 ... NW64.
    Returns the track table and a list of notes about repairs. Known defect of this mirror: the
    first rows of the file (the first 1851 storm) have no Key and no Name. A block of rows without
    a key is assigned the identifier that precedes the next key in the same season
    (`AL{n-1:02d}{year}`) when that identifier is not used elsewhere; otherwise it is dropped. Both
    cases are reported in the notes.
    """
    import pyreadr

    raw = next(iter(pyreadr.read_r(str(path)).values())).copy()
    notes: list[str] = []
    key = raw["Key"].astype(object).where(raw["Key"].notna(), None).tolist()
    name = raw["Name"].astype(object).where(raw["Name"].notna(), None).tolist()
    existing = {k for k in key if k}
    drop = np.zeros(len(raw), dtype=bool)
    i = 0
    while i < len(key):
        if key[i] is not None:
            i += 1
            continue
        j = i
        while j < len(key) and key[j] is None:
            j += 1
        nxt = key[j] if j < len(key) else None
        year = int(pd.Timestamp(raw["DateTime"].iloc[i]).year)
        fixed = None
        if nxt is not None and int(nxt[4:]) == year and int(nxt[2:4]) > 1:
            cand = f"{nxt[:2]}{int(nxt[2:4]) - 1:02d}{year}"
            if cand not in existing:
                fixed = cand
        if fixed:
            for k in range(i, j):
                key[k] = fixed
                name[k] = name[k] or "UNNAMED"
            notes.append(f"rows {i}-{j - 1}: missing Key/Name, assigned {fixed} (precedes {nxt})")
            existing.add(fixed)
        else:
            drop[i:j] = True
            notes.append(f"rows {i}-{j - 1}: missing Key, could not infer an identifier; dropped")
        i = j

    def num(col: str) -> np.ndarray:
        v = pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=float, copy=True)
        v[np.isin(v, _MISSING)] = np.nan
        return v

    out = pd.DataFrame({
        "storm_id": key, "name": name,
        "time": raw["DateTime"].to_numpy(),
        "record": raw["Record"].astype(object).where(raw["Record"].notna(), "").tolist(),
        "status": raw["Status"].astype(str).tolist(),
        "lat": raw["Lat"].to_numpy(dtype=float), "lon": raw["Lon"].to_numpy(dtype=float),
        "vmax_kt": num("Wind"), "mslp_mb": num("Pressure"),
    })
    for r in RADII:
        k, q = r[1:3], r[4:].upper()
        out[r] = num(f"{q}{k}")
    out["rmw_nm"] = np.nan
    out["source"] = "rda-mirror"
    out = out[~drop].reset_index(drop=True)
    out["season"] = [int(s[4:]) for s in out["storm_id"]]
    out, lon_notes = _repair_rda_longitudes(out)
    return _finish(out), notes + lon_notes


def _repair_rda_longitudes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Repair two longitude defects of the rda mirror (found by comparison with the dplyr mirror).

    1. Longitudes of 99.0W and 0.0 were stored as NA (for example Allen 1980-08-10 18 UTC and
       Charley 1986-08-26 12 UTC, which the dplyr mirror gives as -99 and -0). An NA longitude
       whose neighbouring fixes in the same storm are all within 10 degrees of -99 is set to
       -99.0; if they are all within 10 degrees of 0 it is set to 0.0; otherwise it is dropped.
    2. A few longitudes are below -180 (Beulah 1963: -359.0 for 1.0E); they are wrapped into
       (-180, 180] first.
    """
    df = df.sort_values(["storm_id", "time"], kind="stable").reset_index(drop=True)
    notes = []
    lon = df["lon"].to_numpy(dtype=float, copy=True)
    sid = df["storm_id"].to_numpy()
    wrap = np.isfinite(lon) & ((lon <= -180.0) | (lon > 180.0))
    lon[wrap] = (lon[wrap] + 180.0) % 360.0 - 180.0
    fixed, dropped = [], []
    for i in np.nonzero(np.isnan(lon))[0]:
        nb = [lon[k] for k in (i - 1, i + 1)
              if 0 <= k < len(lon) and sid[k] == sid[i] and np.isfinite(lon[k])]
        if nb and all(abs(v + 99.0) <= 10.0 for v in nb):
            lon[i] = -99.0
            fixed.append(i)
        elif nb and all(abs(v) <= 10.0 for v in nb):
            lon[i] = 0.0
            fixed.append(i)
        else:
            dropped.append(i)
    df["lon"] = lon
    if fixed:
        notes.append(f"{len(fixed)} NA longitudes set to -99.0 or 0.0 from their neighbours")
    if dropped:
        keys = ", ".join(f"{sid[i]} {df['time'].iloc[i]}" for i in dropped)
        notes.append(f"{len(dropped)} rows with NA longitude dropped: {keys}")
    if wrap.any():
        notes.append(f"{int(wrap.sum())} longitudes outside (-180, 180] wrapped")
    return df.drop(index=dropped).reset_index(drop=True), notes


def _is_atcf_id(name: str) -> bool:
    """dplyr title-cases names, so an identifier such as AL202011 can appear as "Al202011"."""
    return re.fullmatch(r"AL\d{6}", name.upper()) is not None


_DPLYR_STATUS = {"hurricane": "HU", "tropical storm": "TS", "tropical depression": "TD",
                 "extratropical": "EX", "subtropical depression": "SD",
                 "subtropical storm": "SS", "other low": "LO", "tropical wave": "WV",
                 "disturbance": "DB"}


def load_dplyr_mirror(path: str | pathlib.Path) -> tuple[pd.DataFrame, list[str]]:
    """Load `data-raw/storms.csv` of github.com/tidyverse/dplyr (HURDAT2 1975-2024).

    That file was made from the NHC file hurdat2-1851-2024-040425.txt by data-raw/storms.R, which
    (read on 2026-11-05): keeps years >= 1975; drops every row with missing pressure; keeps only
    the hour of the time (minutes are lost, so a 1905 UTC landfall entry appears as 19 UTC);
    drops the record identifier, the storm identifier and the quadrant wind radii (it keeps only
    two diameters, which are not carried over here); replaces the name UNNAMED by the identifier;
    and takes the year from each row's date, so a storm that crosses 31 December is split.

    Storms are reconstructed as runs of consecutive rows with the same name and no time gap above
    10 days. The identifier is the name when it is an ATCF identifier, otherwise
    `"{NAME}_{season}"` with season the year of the first row (not an ATCF identifier).
    """
    raw = pd.read_csv(path)
    notes: list[str] = []
    t = pd.to_datetime(dict(year=raw["year"], month=raw["month"], day=raw["day"],
                            hour=raw["hour"]))
    new = (raw["name"] != raw["name"].shift()) | (t.diff() > pd.Timedelta(days=10)) | (
        t.diff() < pd.Timedelta(0))
    run = new.cumsum()
    first_year = t.groupby(run).transform("min").dt.year
    ids = []
    for nm, yr in zip(raw["name"], first_year, strict=True):
        ids.append(nm.upper() if _is_atcf_id(nm) else f"{nm.upper()}_{yr}")
    unknown = sorted(set(raw["status"]) - set(_DPLYR_STATUS))
    if unknown:
        raise ValueError(f"unknown status labels {unknown}")
    out = pd.DataFrame({
        "storm_id": ids,
        "name": [nm.upper() if not _is_atcf_id(nm) else "UNNAMED"
                 for nm in raw["name"]],
        "season": first_year.to_numpy(dtype=int), "time": t.to_numpy(),
        "record": pd.array([pd.NA] * len(raw), dtype="string"),
        "status": raw["status"].map(_DPLYR_STATUS),
        "lat": raw["lat"].astype(float), "lon": raw["long"].astype(float),
        "vmax_kt": raw["wind"].astype(float), "mslp_mb": raw["pressure"].astype(float),
    })
    for r in RADII:
        out[r] = np.nan
    out["rmw_nm"] = np.nan
    out["source"] = "dplyr-mirror"
    dup = out.duplicated(["storm_id", "time"], keep=False)
    if dup.any():
        notes.append(f"{int(dup.sum())} rows share storm and hour with another row (minutes were "
                     "dropped by the mirror); all are kept, in file order")
    split = out.groupby("storm_id")["time"].agg(lambda s: s.dt.year.nunique())
    for sid in split[split > 1].index:
        notes.append(f"{sid}: spans two calendar years; rows re-joined into one storm")
    df = _finish(out)
    df["record"] = df["record"].astype("string")
    return df, notes


def synoptic(df: pd.DataFrame) -> pd.DataFrame:
    """Rows at 00, 06, 12, 18 UTC with zero minutes; the first such row per storm and time."""
    t = df["time"]
    keep = (t.dt.hour % 6 == 0) & (t.dt.minute == 0)
    out = df[keep]
    return out[~out.duplicated(["storm_id", "time"], keep="first")]


def compare_sources(rda: pd.DataFrame, dpl: pd.DataFrame,
                    seasons: tuple[int, int] = (1975, 2019)) -> dict:
    """Compare the two mirrors on storms and synoptic times present in both.

    Storms are matched on (name, season) for named storms and on the identifier for unnamed ones.
    Fixes are matched on exact time, using synoptic rows only (the dplyr mirror lost minutes).
    Returns counts and per-field disagreement counts; differences are reported, never removed.
    """
    lo, hi = seasons
    a = synoptic(rda[(rda.season >= lo) & (rda.season <= hi)]).copy()
    b = synoptic(dpl[(dpl.season >= lo) & (dpl.season <= hi)]).copy()
    a["key"] = np.where(a["name"] == "UNNAMED", a["storm_id"], a["name"] + "_" +
                        a["season"].astype(str))
    b["key"] = np.where(b["name"] == "UNNAMED", b["storm_id"], b["name"] + "_" +
                        b["season"].astype(str))
    ka, kb = set(a["key"]), set(b["key"])
    m = a.merge(b, on=["key", "time"], suffixes=("_a", "_b"))
    res = {
        "seasons": f"{lo}-{hi}",
        "storms_rda": len(ka), "storms_dplyr": len(kb), "storms_both": len(ka & kb),
        "storms_only_rda": sorted(ka - kb), "storms_only_dplyr": sorted(kb - ka),
        "fixes_rda": len(a), "fixes_dplyr": len(b), "fixes_matched": len(m),
        "fixes_rda_missing_pressure": int(a["mslp_mb"].isna().sum()),
    }
    only_a = a[a["key"].isin(ka - kb)].groupby("key")
    res["storms_only_rda_all_pressure_missing"] = int(only_a["mslp_mb"].apply(
        lambda s: s.isna().all()).sum())
    # dplyr drops rows without pressure first, then storms without a TD/TS/HU row.
    res["storms_only_rda_no_td_ts_hu"] = int(a[a["key"].isin(ka - kb)].groupby("key").apply(
        lambda g: g["mslp_mb"].notna().any()
        and not g.loc[g["mslp_mb"].notna(), "status"].isin(["TD", "TS", "HU"]).any()).sum())
    common = a[a["key"].isin(ka & kb)]
    res["fixes_rda_unmatched_in_common_storms"] = int(len(common) - len(m))
    for col, tol in (("lat", 1e-9), ("lon", 1e-9), ("vmax_kt", 1e-9), ("mslp_mb", 1e-9)):
        x, y = m[f"{col}_a"], m[f"{col}_b"]
        both = x.notna() & y.notna()
        res[f"diff_{col}"] = int((both & ((x - y).abs() > tol)).sum())
        res[f"maxabs_{col}"] = float((x - y)[both].abs().max()) if both.any() else 0.0
    res["diff_status"] = int((m["status_a"] != m["status_b"]).sum())
    rows = []
    for col in ("lat", "lon", "vmax_kt", "mslp_mb", "status"):
        x, y = m[f"{col}_a"], m[f"{col}_b"]
        bad = (x != y) & x.notna() & y.notna()
        for _, r in m[bad].iterrows():
            rows.append((r["key"], str(r["time"]), col, r[f"{col}_a"], r[f"{col}_b"]))
    res["differences"] = rows
    return res
