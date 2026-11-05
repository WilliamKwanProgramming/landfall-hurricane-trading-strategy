"""Pieces of the real-data study and of the synthetic simulator."""

import numpy as np
import pandas as pd
import pytest
from test_forecast import synthetic_tracks

from landfall import forecast as F
from landfall import realstudy as rs
from landfall import signals
from landfall.geo import LandGrid
from landfall.study import eventstudy as es
from landfall.study import simulate as sim


def test_drop_overlapping_rule():
    keep = rs.drop_overlapping(np.array([10, 20, 26, 40, 41]))
    np.testing.assert_array_equal(keep, [True, False, True, False, False])
    keep = rs.drop_overlapping(np.array([41, 10, 26]))  # order of input does not matter
    np.testing.assert_array_equal(keep, [False, True, True])


def test_quality_screen():
    r = np.random.default_rng(0).normal(0, 0.01, 400)
    assert rs.quality_ok(r, 350)
    r2 = r.copy()
    r2[350 - 280: 350 - 280 + 26] = 0.0  # 26 of 250 estimation days unchanged: > 10%
    assert not rs.quality_ok(r2, 350)
    r3 = r.copy()
    r3[345: 347] = 0.0  # 2 of 16 event days: 12.5%
    assert not rs.quality_ok(r3, 350)
    r4 = r.copy()
    r4[345] = 0.0  # 1 of 16: 6%
    assert rs.quality_ok(r4, 350)


def test_placebo_dates_respect_the_design():
    dates = pd.bdate_range("1990-01-01", "2022-12-30")
    events = pd.Series(pd.to_datetime(["2005-08-29 11:00", "2017-08-26 03:00"]))
    landfalls = pd.Series(pd.to_datetime(["2005-08-29", "2017-08-26", "2008-09-01",
                                          "2012-08-29"]))
    rows = rs.placebo_day0(events, landfalls, dates, (1991, 2022), 200,
                           np.random.default_rng(1))
    assert (rows >= 0).all()
    for j, t in enumerate(events):
        d = dates[rows[:, j]]
        assert (d.year != t.year).all()
        # Day 0 is the first trading day at or after a day of the same ISO week (weekend days
        # roll to the following Monday, i.e. at most into the next week).
        wk = np.array([x.isocalendar().week for x in d])
        assert np.isin(wk, [t.isocalendar().week, t.isocalendar().week + 1]).all()
        for lf in landfalls:
            assert (np.abs((d - lf).days) > 21 - 3).all()
    again = rs.placebo_day0(events, landfalls, dates, (1991, 2022), 200,
                            np.random.default_rng(1))
    np.testing.assert_array_equal(rows, again)


def test_placebo_position_by_hand():
    le, p = rs.placebo_position(0.3, np.array([0.1, 0.2, 0.3, 0.4]))
    assert le == 0.75 and p == 1.0
    le, p = rs.placebo_position(1.0, np.arange(100) / 100)
    assert le == 1.0 and p == 0.0


def test_entry_rows_use_the_first_close_after_availability():
    dates = pd.DatetimeIndex(["2005-08-26", "2005-08-29"])
    rows = rs.entry_rows_from_times(dates, pd.Series(pd.to_datetime(
        ["2005-08-26 15:00", "2005-08-26 21:00"])))
    np.testing.assert_array_equal(rows, [0, 1])


GRID = LandGrid(20.0, -90.0, 0.1, 0.1, np.zeros((100, 200), dtype=np.uint8))
GRID.land[:, 100:] = 1


def test_landfall_probability_lookahead_guard():
    tracks = synthetic_tracks(seed=21)
    smp = F.build_samples(tracks)
    base = signals.landfall_probabilities(smp, range(1998, 1999), GRID, n_draws=100, seed=3)
    t = base["time"].iloc[len(base) // 2]
    later = tracks["time"] > t
    altered = tracks.copy()
    altered.loc[later, "lon"] += 5.0
    for tr in (altered, tracks[~later]):
        p2 = signals.landfall_probabilities(F.build_samples(tr), range(1998, 1999), GRID,
                                            n_draws=100, seed=3)
        a = base[base["time"] <= t].set_index(["storm_id", "time"])["p_us_landfall"]
        b = p2[p2["time"] <= t].set_index(["storm_id", "time"])["p_us_landfall"]
        pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


def test_first_trigger_is_the_earliest_crossing():
    probs = pd.DataFrame({"storm_id": ["A"] * 4 + ["B"] * 2,
                          "time": pd.to_datetime(["2000-09-01 00:00", "2000-09-01 06:00",
                                                  "2000-09-01 12:00", "2000-09-01 18:00",
                                                  "2000-09-02 00:00", "2000-09-03 00:00"]),
                          "p_us_landfall": [0.1, 0.5, 0.2, 0.9, 0.3, 0.35]})
    trig = signals.first_triggers(probs, 0.4)
    assert trig["storm_id"].tolist() == ["A"]
    assert trig["time"].item() == pd.Timestamp("2000-09-01 06:00")


# ---- simulator ------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(sim.SCENARIOS))
def test_injected_abnormal_return_is_exact(name):
    sc = sim.SCENARIOS[name]
    a = sim.simulate_panel(sc, 5, 3, 0.0, np.random.default_rng(9), n_reps=2)
    b = sim.simulate_panel(sc, 5, 3, 0.013, np.random.default_rng(9), n_reps=2)
    np.testing.assert_array_equal(a[0], b[0])  # estimation windows untouched
    ma = es.fit_market_model(*(x.reshape(-1, x.shape[-1]) for x in a[:4]))
    mb = es.fit_market_model(*(x.reshape(-1, x.shape[-1]) for x in b[:4]))
    np.testing.assert_allclose(mb.car(0, 10) - ma.car(0, 10), 0.013, atol=1e-12)
    np.testing.assert_allclose(mb.car(-5, -1), ma.car(-5, -1), atol=1e-15)


def test_garch_noise_has_the_stated_variance():
    sc = sim.SCENARIOS["garch_t"]
    x = sim.noise((400, 2000), 0.02, sc, np.random.default_rng(10))
    assert x.std() == pytest.approx(0.02, rel=0.03)
    # Volatility clustering: squared returns are autocorrelated.
    sq = x ** 2 - (x ** 2).mean()
    ac = (sq[:, 1:] * sq[:, :-1]).mean() / (sq ** 2).mean()
    assert ac > 0.05


@pytest.mark.parametrize("sc,tol", [(sim.Scenario(False, None, 1.0, 0.3), 0.02),
                                    (sim.SCENARIOS["garch_t_cross_correlation"], 0.05)])
def test_cross_correlation_scenario(sc, tol):
    est_r, est_m, win_r, win_m, groups = sim.simulate_panel(sc, 400, 2, 0.0,
                                                            np.random.default_rng(11))
    mm = es.fit_market_model(est_r[0], est_m[0], win_r[0], win_m[0])
    pairs = mm.resid_est.reshape(400, 2, -1)
    # Unconditional correlation, pooled over events. Under GARCH-t the estimator is noisy
    # (heavy tails), hence the wider tolerance.
    c = np.corrcoef(pairs[:, 0].ravel(), pairs[:, 1].ravel())[0, 1]
    assert c == pytest.approx(0.3, abs=tol)
