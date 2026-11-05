"""Power of the per-asset tests: grid interpolation, the t-test cross-check, the simulation."""

import numpy as np
import pandas as pd
import pytest

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
