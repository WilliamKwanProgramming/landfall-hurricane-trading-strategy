"""Power of the per-asset tests: grid interpolation, the t-test cross-check, the simulation."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from landfall import pipeline as pl
from landfall.study import eventstudy as es
from landfall.study import power as pw
from landfall.study import simulate as sim

# ---- interpolation of the 80% point ---------------------------------------------------------


def test_interpolate_target_is_linear_between_grid_points():
    d, p = [0.0, 1.0, 2.0, 3.0], [0.05, 0.4, 0.9, 1.0]
    assert pw.interpolate_target(d, p, 0.8) == pytest.approx(1.0 + 0.4 / 0.5)
    assert pw.interpolate_target(d, p, 0.99) == pytest.approx(2.0 + 0.09 / 0.1)
    assert pw.interpolate_target(d, p, 0.9) == pytest.approx(2.0)  # a grid point itself


def test_interpolate_target_edge_cases():
    d = [0.0, 1.0, 2.0]
    assert np.isnan(pw.interpolate_target(d, [0.05, 0.3, 0.6], 0.8))  # never reached
    assert pw.interpolate_target(d, [0.9, 0.95, 1.0], 0.8) == 0.0  # already reached at the start
    # A curve that is not monotone: the first crossing counts.
    assert pw.interpolate_target(d, [0.05, 0.9, 0.5], 0.8) == pytest.approx(0.75 / 0.85)


# ---- plug-in t-test power -------------------------------------------------------------------


def test_plugin_t_power_known_values():
    # No effect: the size of the test.
    assert pw.plugin_t_power(0.01, 30, 0.0) == pytest.approx(0.05, abs=1e-12)
    # With many degrees of freedom the t is normal: an effect of 1.96 standard errors is
    # detected with probability 1/2 (plus the negligible far tail).
    assert pw.plugin_t_power(0.01, 100_000, 0.0196) == pytest.approx(0.5, abs=2e-4)
    # An effect of 2.8 standard errors gives the textbook 80%.
    assert pw.plugin_t_power(0.01, 100_000, 0.028) == pytest.approx(0.80, abs=0.005)


def test_plugin_t_power_is_symmetric_and_increasing():
    grid = [0.0, 0.005, 0.01, 0.02, 0.04]
    p = [pw.plugin_t_power(0.01, 35, x) for x in grid]
    assert all(b > a for a, b in zip(p, p[1:], strict=False))
    assert pw.plugin_t_power(0.01, 35, -0.02) == pytest.approx(pw.plugin_t_power(0.01, 35, 0.02))
    # More noise (larger standard error) means less power at the same effect.
    assert pw.plugin_t_power(0.02, 35, 0.02) < pw.plugin_t_power(0.01, 35, 0.02)


# ---- the simulated power curve --------------------------------------------------------------

REPS = 300


def test_power_curve_has_size_near_5_percent_and_power_near_one():
    # 30 events, 1% residual sd: the sd of a CAR[0, +10] is about 3.3%, its mean over 30 events
    # has a standard error of about 0.6%, so a true 1.2% is detected about half the time.
    c = pw.power_curve(30, 0.01, [0.0, 0.012, 0.06], REPS, [1, 2, 3], scenario="gaussian")
    assert list(c["delta"]) == [0.0, 0.012, 0.06]
    assert (c["reps"] == REPS).all()
    for t in sim.TESTS:
        assert 0.01 <= c.loc[0, f"reject_{t}"] <= 0.11, t  # size
        assert c.loc[2, f"reject_{t}"] > 0.9, t  # a 6% effect is certain to be found
    assert 0.3 < c.loc[1, "reject_bmp"] < 0.7
    p = c.loc[1, "reject_bmp"]
    assert c.loc[1, "reject_bmp_mcse"] == pytest.approx(np.sqrt(p * (1 - p) / REPS))
    assert c["reject_bmp"].is_monotonic_increasing


def test_power_curve_uses_the_requested_residual_volatility():
    quiet = pw.power_curve(30, 0.01, [0.03], REPS, [4], scenario="gaussian")
    noisy = pw.power_curve(30, 0.04, [0.03], REPS, [4], scenario="gaussian")
    assert quiet.loc[0, "reject_bmp"] > 0.97
    assert noisy.loc[0, "reject_bmp"] < 0.5  # the same effect is 4 times harder to see


def test_power_curve_is_reproducible_and_depends_on_the_key():
    a = pw.power_curve(20, 0.01, [0.0, 0.02], 50, [7, 5, 0])
    b = pw.power_curve(20, 0.01, [0.0, 0.02], 50, [7, 5, 0])
    c = pw.power_curve(20, 0.01, [0.0, 0.02], 50, [7, 5, 1])
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_each_delta_has_its_own_random_stream():
    """Delta number k is simulated with default_rng([*key, k]), whatever else is on the grid."""
    grid = pw.power_curve(20, 0.01, [0.0, 0.01, 0.02], 40, [7, 5])
    for k, delta in enumerate([0.0, 0.01, 0.02]):
        alone = pw.power_curve(20, 0.01, [delta], 40, [7, 5, k])
        pd.testing.assert_frame_equal(alone, grid.iloc[[k]].reset_index(drop=True))


# ---- the simulator's per-call residual volatility -------------------------------------------


@pytest.mark.parametrize("sigma", [0.005, 0.02, 0.05])
def test_simulated_market_model_residual_sd_equals_the_requested_one(sigma):
    sc = sim.SCENARIOS["gaussian"]
    est_r, est_m, win_r, win_m, _ = sim.simulate_panel(sc, 400, 1, 0.0, np.random.default_rng(12),
                                                       sigma_e=sigma)
    mm = es.fit_market_model(est_r[0], est_m[0], win_r[0], win_m[0])
    assert np.sqrt(np.mean(mm.sigma ** 2)) == pytest.approx(sigma, rel=0.02)


def test_default_residual_sd_is_unchanged_by_the_new_argument():
    sc = sim.SCENARIOS["garch_t"]
    a = sim.simulate_panel(sc, 6, 2, 0.01, np.random.default_rng(13), n_reps=2)
    b = sim.simulate_panel(sc, 6, 2, 0.01, np.random.default_rng(13), n_reps=2,
                           sigma_e=sim.SIGMA_E)
    for x, y in zip(a[:4], b[:4], strict=True):
        np.testing.assert_array_equal(x, y)


def test_no_bootstrap_is_drawn_when_n_boot_is_zero():
    sc = sim.SCENARIOS["gaussian"]
    rows = sim.replicate(sc, 10, 1, 0.0, np.random.default_rng(14), n_reps=3, n_boot=0)
    assert len(rows) == 3
    assert all(np.isnan(r["boot_cover"]) for r in rows)
    rows = sim.replicate(sc, 10, 1, 0.0, np.random.default_rng(14), n_reps=2, n_boot=25)
    assert all(r["boot_cover"] in (True, False) for r in rows)


# ---- power at each asset's own events and volatility ----------------------------------------


def _fit(sigmas, group="storm"):
    """The two things power_by_asset reads from a fit: the residual sds and the asset group."""
    return SimpleNamespace(mm=SimpleNamespace(sigma=np.asarray(sigmas, dtype=float)),
                           data=SimpleNamespace(asset=SimpleNamespace(group=group)))


GRID = (0.0, 0.01, 0.02, 0.03, 0.06)


def test_power_by_asset_uses_the_event_count_and_the_rms_residual_sd():
    fits = {"A": _fit([0.01] * 10 + [0.03] * 10)}
    curves, summary = pl.power_by_asset(fits, GRID, 20, [3])
    assert summary["events"].tolist() == [20]
    assert summary["resid_sd_rms"].iloc[0] == pytest.approx(np.sqrt(5e-4))  # not the mean 0.02
    assert summary["group"].tolist() == ["storm"]
    assert curves["delta"].tolist() == list(GRID) and (curves["events"] == 20).all()


def test_power_by_asset_skips_assets_without_a_fit_but_counts_their_position():
    fits = {"A": _fit([0.012] * 25), "B": None, "C": _fit([0.02] * 30, "control")}
    got: list[dict] = []
    curves, summary = pl.power_by_asset(fits, GRID, 20, [3], got.append)
    assert summary["asset"].tolist() == ["A", "C"] and [g["asset"] for g in got] == ["A", "C"]
    # C is asset number 2: its random key is [3, 2] whatever B is.
    direct = pw.power_curve(30, 0.02, GRID, 20, [3, 2])
    c = summary.set_index("asset").loc["C"]
    assert c["size_bmp"] == direct["reject_bmp"][0]
    assert c["power_bmp_1pct"] == direct["reject_bmp"][1]
    assert c["power_bmp_2pct"] == direct["reject_bmp"][2]
    assert c["power_bmp_3pct"] == direct["reject_bmp"][3]
    np.testing.assert_array_equal(curves[curves["asset"] == "C"]["reject_bmp"],
                                  direct["reject_bmp"])


def test_power_by_asset_needs_the_reporting_points_on_the_grid():
    with pytest.raises(ValueError, match="0.03"):
        pl.power_by_asset({"A": _fit([0.01] * 20)}, (0.0, 0.01, 0.02, 0.05), 5, [1])


def test_power_by_asset_finds_the_80_percent_point_between_grid_points():
    # Residual sd 1%, 35 events: 80% power at a true CAR of about 1.7%, between 1% and 2%.
    _, summary = pl.power_by_asset({"A": _fit([0.01] * 35)}, GRID, 200, [8])
    d80 = summary["delta_80pct_power_bmp"].iloc[0]
    assert 0.010 < d80 < 0.025
    assert summary["power_bmp_3pct"].iloc[0] > 0.95 > 0.5 > summary["size_bmp"].iloc[0]
