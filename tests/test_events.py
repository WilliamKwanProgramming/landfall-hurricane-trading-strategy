"""Landfall definitions on synthetic tracks and on four well-known US landfalls."""

import numpy as np
import pandas as pd
import pytest

from landfall import events as ev
from landfall.data.hurdat2 import TRACK_COLUMNS
from landfall.geo import LandGrid


def _track(lats, lons, winds, start="2000-09-01 00:00", sid="AL012000", record=None):
    n = len(lats)
    df = pd.DataFrame({c: np.nan for c in TRACK_COLUMNS}, index=range(n))
    df["storm_id"], df["name"], df["season"] = sid, "TEST", 2000
    df["time"] = pd.date_range(start, periods=n, freq="6h").astype("datetime64[s]")
    df["record"] = record if record is not None else [""] * n
    df["status"] = "HU"
    df["lat"], df["lon"], df["vmax_kt"] = lats, lons, winds
    df["source"] = "test"
    return df


# Land is everything east of 80W between 20N and 40N (a straight north-south coastline).
GRID = LandGrid(20.0, -90.0, 0.01, 0.01, np.zeros((2000, 2000), dtype=np.uint8))
GRID.land[:, 1000:] = 1


def test_straight_crossing_time_and_place():
    # Moves east 1 degree per 6 h along 30N; crosses 80W one quarter of the way through the
    # segment that starts at 80.25W, i.e. 1.5 h after its start.
    tr = _track([30.0] * 4, [-81.25, -80.25, -79.25, -78.25], [100, 90, 60, 40])
    lf = ev.geometric_landfalls(tr, GRID, step_minutes=10)
    assert len(lf) == 1
    assert lf.loc[0, "time"] == pd.Timestamp("2000-09-01 07:30")
    assert lf.loc[0, "lon"] == pytest.approx(-80.0)
    assert lf.loc[0, "vmax_kt"] == pytest.approx(90 - 30 * 0.25)


def test_storm_starting_over_land_is_not_a_landfall():
    tr = _track([30.0] * 3, [-79.0, -78.0, -77.0], [40, 40, 40])
    assert ev.geometric_landfalls(tr, GRID).empty


def test_two_crossings_and_minimum_sea_time():
    # 16 h at sea, landfall, back to sea for 6 h, second landfall; with a 12-h minimum sea time
    # only the first counts.
    tr = _track([30.0] * 6, [-84.0, -83.0, -82.0, -79.0, -81.0, -79.0], [80] * 6)
    assert len(ev.geometric_landfalls(tr, GRID, min_sea_hours=1.0)) == 2
    assert len(ev.geometric_landfalls(tr, GRID, min_sea_hours=12.0)) == 1


def test_non_synoptic_fixes_are_ignored():
    tr = _track([30.0] * 3, [-81.5, -80.5, -79.5], [80] * 3)
    extra = tr.iloc[[1]].copy()
    extra["time"] = pd.Timestamp("2000-09-01 07:00")
    extra["lon"] = 10.0  # absurd position: must not be used (not a synoptic fix)
    both = pd.concat([tr, extra]).sort_values("time")
    a = ev.geometric_landfalls(tr, GRID)
    b = ev.geometric_landfalls(both, GRID)
    pd.testing.assert_frame_equal(a, b)


def test_flagged_landfalls():
    tr = _track([30.0] * 3, [-81.0, -80.0, -79.0], [80, 70, 50], record=["", "L", ""])
    lf = ev.flagged_landfalls(tr)
    assert len(lf) == 1 and lf.loc[0, "time"] == pd.Timestamp("2000-09-01 06:00")


def test_polygon_against_matplotlib():
    from matplotlib.path import Path

    rng = np.random.default_rng(3)
    lat = rng.uniform(20, 50, 5000)
    lon = rng.uniform(-105, -60, 5000)
    ours = ev.in_polygon(lat, lon)
    ref = Path(np.array(ev.US_COAST_POLYGON)).contains_points(np.c_[lon, lat])
    assert np.mean(ours == ref) == 1.0


@pytest.mark.parametrize("name,lat,lon,inside", [
    ("Galveston", 29.3, -94.8, True), ("Miami", 25.77, -80.19, True),
    ("Key West", 24.55, -81.8, True), ("Cape Hatteras", 35.25, -75.53, True),
    ("Montauk", 41.07, -71.85, True), ("South Padre Island", 26.1, -97.16, True),
    ("Bimini", 25.73, -79.28, False), ("Grand Bahama", 26.6, -78.6, False),
    ("Havana", 23.13, -82.38, False), ("Mexican coast 25.5N", 25.5, -97.4, False),
    ("Tampico", 22.25, -97.85, False), ("Yarmouth NS", 43.84, -66.12, False),
    ("Bermuda", 32.3, -64.78, False),
])
def test_polygon_known_places(name, lat, lon, inside):
    assert bool(ev.in_polygon(lat, lon)[0]) is inside, name


def test_dispersion_index():
    r = ev.dispersion_test(np.array([1, 1, 1, 1]))
    assert r["D"] == 0.0 and r["df"] == 3
    x = np.array([0, 2, 4, 2])
    r = ev.dispersion_test(x)
    assert r["D"] == pytest.approx(((x - 2) ** 2).sum() / 2)
    rng = np.random.default_rng(0)
    p = [ev.dispersion_test(rng.poisson(2.0, 100))["p_two_sided"] for _ in range(400)]
    assert 0.02 < np.mean(np.array(p) < 0.05) < 0.09  # close to nominal size


# ---- real tracks: times from the NHC tropical cyclone reports, positions from HURDAT2 -----

KNOWN = [  # storm, NHC report time (UTC), lat, lon
    ("AL041992", "1992-08-24 09:05", 25.5, -80.3),   # Andrew, Homestead FL
    ("AL041992", "1992-08-26 08:30", 29.6, -91.5),   # Andrew, Point Chevreuil LA
    ("AL122005", "2005-08-25 22:30", 26.0, -80.1),   # Katrina, Miami-Dade/Broward border FL
    ("AL122005", "2005-08-29 11:10", 29.3, -89.6),   # Katrina, Buras LA
    ("AL122005", "2005-08-29 14:45", 30.2, -89.6),   # Katrina, Louisiana/Mississippi border
    ("AL092017", "2017-08-26 03:00", 28.0, -96.9),   # Harvey, San Jose Island TX
    ("AL092017", "2017-08-26 06:00", 28.2, -97.1),   # Harvey, Holiday Beach TX
    ("AL092017", "2017-08-30 08:00", 29.8, -93.5),   # Harvey, near Cameron LA
]
IAN = [("2022-09-28 19:05", 26.7, -82.2), ("2022-09-30 18:05", 33.3, -79.2)]


@pytest.mark.parametrize("sid,when,lat,lon", KNOWN)
def test_known_landfalls_flagged(rda_tracks, sid, when, lat, lon):
    lf = ev.flagged_landfalls(rda_tracks[rda_tracks.storm_id == sid])
    hit = lf[lf["time"] == pd.Timestamp(when)]
    assert len(hit) == 1
    assert hit["lat"].item() == lat and hit["lon"].item() == lon
    assert bool(ev.in_polygon(lat, lon)[0])


@pytest.mark.parametrize("sid,when,lat,lon", KNOWN)
def test_known_landfalls_geometric(rda_tracks, land_grid, sid, when, lat, lon):
    lf = ev.us_landfalls(ev.geometric_landfalls(rda_tracks[rda_tracks.storm_id == sid],
                                                land_grid))
    dt = (lf["time"] - pd.Timestamp(when)).abs()
    k = dt.idxmin()
    assert dt[k] <= pd.Timedelta(hours=2)
    from landfall.geo import haversine_km

    assert haversine_km(lat, lon, lf.loc[k, "lat"], lf.loc[k, "lon"]) < 50


def test_ian_2022_geometric_on_dplyr_mirror(dplyr_tracks, land_grid):
    lf = ev.us_landfalls(ev.geometric_landfalls(dplyr_tracks[dplyr_tracks.storm_id == "IAN_2022"],
                                                land_grid))
    from landfall.geo import haversine_km

    for when, lat, lon in IAN:
        dt = (lf["time"] - pd.Timestamp(when)).abs()
        k = dt.idxmin()
        assert dt[k] <= pd.Timedelta(hours=1)
        assert haversine_km(lat, lon, lf.loc[k, "lat"], lf.loc[k, "lon"]) < 50
    assert lf.loc[0, "vmax_kt"] >= 113  # category 4 at the Florida landfall
