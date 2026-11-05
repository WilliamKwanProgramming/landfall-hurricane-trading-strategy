"""Exposure scores: damage function, bundled maps, and the look-ahead guard."""

import numpy as np
import pandas as pd
import pytest
from conftest import cpp_module
from test_forecast import synthetic_tracks

from landfall import exposure as ex
from landfall import footprint as fp
from landfall import forecast as F
from landfall.geo import LandGrid


def test_emanuel_damage_by_hand():
    assert ex.emanuel_damage(50.0) == 0.0 and ex.emanuel_damage(20.0) == 0.0
    assert ex.emanuel_damage(150.0) == pytest.approx(0.5)
    assert ex.emanuel_damage(100.0) == pytest.approx(0.125 / 1.125)
    v = ex.emanuel_damage(np.linspace(0, 300, 301))
    assert (np.diff(v) >= 0).all() and v.max() < 1.0


@pytest.mark.parametrize("name", ["florida_weighted", "gulf_energy_weighted"])
def test_bundled_maps_are_labelled_and_valid(name):
    m = ex.load_exposure_map(name)
    assert m["weight"].sum() == pytest.approx(1.0)
    text = (ex.MAPS_DIR / f"{name}.csv").read_text()
    assert "ILLUSTRATIVE ASSUMPTION" in text and "NOT derived" in text


def test_map_validation(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("point,lat,lon,weight\na,25,-80,0.7\nb,26,-81,0.7\n")
    with pytest.raises(ValueError):
        ex.load_exposure_map(p)


def test_storm_structure_choices():
    rm_w, b = fp.storm_structure(100.0, 950.0, 25.0)
    assert rm_w == pytest.approx(float(fp.rmax_willoughby_km(100.0, 25.0)))
    assert b == pytest.approx(float(fp.holland_b(100.0, 950.0)))
    rm_r, _ = fp.storm_structure(100.0, 950.0, 25.0, [120, 100, 0, np.nan])
    r34 = 110 * 1.852
    assert fp.symmetric_profile(r34, 100.0, rm_r, b) == pytest.approx(34.0)


GRID = LandGrid(20.0, -90.0, 0.1, 0.1, np.zeros((100, 200), dtype=np.uint8))
GRID.land[:, 100:] = 1  # land east of 80W


def _forecast_and_model(tracks, season, k=3):
    smp = F.build_samples(tracks)
    m = F.fit_model(smp, season)
    row = smp[smp.season == season].iloc[k]
    pred = {key: v[0] for key, v in m.predict_targets(smp.loc[[row.name]]).items()}
    return F.to_track_forecast(row, pred), m


def test_score_and_landfall_probability_simple_case():
    # A forecast heading straight at the coast with no error: certain US landfall
    # (the land is inside the US polygon only near 80W/25-30N), and higher scores on the coast.
    from landfall.data.atcf import TrackForecast

    fc = TrackForecast("X", pd.Timestamp("2000-09-01"), pd.Timestamp("2000-09-01"),
                       np.array([0.0, 12, 24, 48, 72, 96, 120]), np.full(7, 27.0),
                       np.array([-84.0, -82.5, -81.0, -78.0, -75.0, -72.0, -69.0]),
                       np.full(7, 120.0), "test")
    emap = pd.DataFrame({"point": ["coast", "far"], "lat": [27.0, 22.0],
                         "lon": [-80.1, -86.0], "weight": [0.5, 0.5]})
    r = ex.forecast_exposure(fc, np.zeros((1, 6, 3)), np.zeros(10, int), 30.0, 1.4, emap, GRID)
    assert r.p_us_landfall == 1.0
    assert r.p_ge_64[0] == 1.0 and r.p_ge_64[1] == 0.0
    assert r.score_se < 1e-15 and 0 < r.score < 0.5


def test_exposure_lookahead_guard():
    """The score at t is unchanged when any track data after t is altered or deleted."""
    tracks = synthetic_tracks(seed=11)
    emap = ex.load_exposure_map("florida_weighted")
    season = 1999
    fc, m = _forecast_and_model(tracks, season)
    t = fc.issue_time
    draws = np.random.default_rng(0).integers(0, len(m.residuals), 64)
    base = ex.forecast_exposure(fc, m.residuals, draws, 30.0, 1.3, emap, GRID)

    later = tracks["time"] > t
    altered = tracks.copy()
    altered.loc[later, "lon"] -= 4.0
    altered.loc[later, "vmax_kt"] *= 2
    for tr in (altered, tracks[~later]):
        fc2, m2 = _forecast_and_model(tr, season)
        assert fc2.issue_time == t
        r2 = ex.forecast_exposure(fc2, m2.residuals, draws, 30.0, 1.3, emap, GRID)
        assert r2.score == base.score and r2.p_us_landfall == base.p_us_landfall


def test_cpp_engine_gives_the_same_score():
    cpp_module()
    tracks = synthetic_tracks(seed=12)
    fc, m = _forecast_and_model(tracks, 1998)
    draws = np.random.default_rng(1).integers(0, len(m.residuals), 100)
    emap = ex.load_exposure_map("gulf_energy_weighted")
    a = ex.forecast_exposure(fc, m.residuals, draws, 40.0, 1.2, emap, GRID, engine="numpy")
    b = ex.forecast_exposure(fc, m.residuals, draws, 40.0, 1.2, emap, GRID, engine="cpp")
    assert b.score == pytest.approx(a.score, rel=1e-9, abs=1e-15)
