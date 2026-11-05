"""Event-study statistics: closed forms, hand calculations and independent implementations."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from landfall import pipeline as pl
from landfall.study import backtest as bt
from landfall.study import bootstrap as bs
from landfall.study import eventstudy as es
from landfall.study import inference as inf
from landfall.study import multiple as mt
from landfall.study import sharpe as sh

# ---- market model ---------------------------------------------------------------------------


def test_market_model_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.01, (3, 250))
    r = 0.0002 + np.array([[0.8], [1.1], [1.5]]) * m + rng.normal(0, 0.02, (3, 250))
    wm = rng.normal(0, 0.01, (3, 16))
    wr = rng.normal(0, 0.02, (3, 16))
    mm = es.fit_market_model(r, m, wr, wm)
    for i in range(3):
        fit = sm.OLS(r[i], sm.add_constant(m[i])).fit()
        assert mm.alpha[i] == pytest.approx(fit.params[0], rel=1e-10)
        assert mm.beta[i] == pytest.approx(fit.params[1], rel=1e-10)
        assert mm.sigma[i] == pytest.approx(np.sqrt(fit.scale), rel=1e-10)  # T - 2 dof
        np.testing.assert_allclose(mm.ar[i], wr[i] - fit.params[0] - fit.params[1] * wm[i],
                                   rtol=1e-10)


def test_car_variance_and_scar_distribution_by_simulation():
    """With fixed market returns and normal errors, Var(CAR) matches the formula and SCAR is
    t(T - 2) distributed: E[SCAR^2] = (T - 2) / (T - 4)."""
    rng = np.random.default_rng(1)
    t_est, reps = 30, 40000
    m_est = rng.normal(0.001, 0.012, t_est)
    m_win = rng.normal(0.004, 0.012, 16)
    sig = 0.02
    est_r = 0.5 * m_est + rng.normal(0, sig, (reps, t_est))
    win_r = 0.5 * m_win + rng.normal(0, sig, (reps, 16))
    mm = es.fit_market_model(est_r, np.tile(m_est, (reps, 1)), win_r, np.tile(m_win, (reps, 1)))
    car = mm.car(0, 10)
    d = 11
    dm = (m_win[5:16] - m_est.mean()).sum()
    v_true = sig ** 2 * (d + d ** 2 / t_est + dm ** 2 / ((m_est - m_est.mean()) ** 2).sum())
    assert car.var() == pytest.approx(v_true, rel=0.03)
    scar = mm.scar(0, 10)
    assert (scar ** 2).mean() == pytest.approx((t_est - 2) / (t_est - 4), rel=0.03)
    assert stats.kstest(scar, stats.t(t_est - 2).cdf).pvalue > 0.01


def test_car_windows_and_day_index():
    ar = np.arange(16, dtype=float)[None, :]  # AR on day k equals k + 5
    mm = es.MarketModel(np.zeros(1), np.zeros(1), np.ones(1), 250, np.zeros(1), np.ones(1),
                        np.zeros((1, 250)), ar, np.zeros((1, 16)), (-5, 10))
    assert mm.car(0, 0)[0] == 5.0
    assert mm.car(-5, -1)[0] == sum(range(0, 5))
    assert mm.car(0, 10)[0] == sum(range(5, 16))
    with pytest.raises(ValueError):
        mm.idx(11)


def test_event_rows_close_rule():
    dates = pd.DatetimeIndex(["2005-08-26", "2005-08-29", "2005-08-30"])  # Fri, Mon, Tue
    t = ["2005-08-26 19:59", "2005-08-26 20:01", "2005-08-28 12:00", "2005-08-29 11:10",
         "2005-08-31 00:00"]
    np.testing.assert_array_equal(es.event_rows(dates, t), [0, 1, 1, 1, -1])


def test_extract_windows_do_not_overlap():
    n = 400
    r = np.arange(n, dtype=float)
    est_r, est_m, win_r, win_m = es.extract(r, r.copy(), np.array([300]))
    assert est_r[0, 0] == 300 - 30 - 250 and est_r[0, -1] == 300 - 31
    assert win_r[0, 0] == 295 and win_r[0, -1] == 310
    assert est_r.max() < win_r.min()
    with pytest.raises(ValueError):
        es.extract(r, r.copy(), np.array([200]))


# ---- test statistics ------------------------------------------------------------------------


def test_patell_bmp_naive_by_hand():
    scar = np.array([1.0, 2.0, -0.5, 0.5])
    p = inf.patell(scar, 250)
    assert p.stat == pytest.approx(3.0 / np.sqrt(4 * 248 / 246))
    b = inf.bmp(scar)
    assert b.stat == pytest.approx(0.75 / (np.std(scar, ddof=1) / 2))
    assert b.p == pytest.approx(2 * stats.t.sf(b.stat, 3))
    nt = inf.naive_t(scar)
    assert nt.stat == pytest.approx(stats.ttest_1samp(scar, 0).statistic)


def test_kp_adjustment_and_average_correlation():
    assert inf.kp_factor(0.0, 50) == 1.0
    assert inf.kp_factor(0.1, 11) == pytest.approx(np.sqrt(0.9 / 2.0))
    rng = np.random.default_rng(2)
    common = rng.normal(size=500)
    x = np.vstack([common + rng.normal(size=500) for _ in range(3)]
                  + [rng.normal(size=500) for _ in range(2)])
    groups = np.array([0, 0, 0, 1, 2])
    c = np.corrcoef(x[:3])
    pairs_same = (c.sum() - 3) / 2
    assert inf.average_cross_correlation(x, groups) == pytest.approx(pairs_same / 10)


def test_rank_test_by_hand_and_against_brute_force():
    rng = np.random.default_rng(3)
    est = rng.normal(size=(6, 20))
    win = rng.normal(size=(6, 5))
    r = inf.rank_test(est, win, slice(1, 4))
    full = np.hstack([est, win])
    L = full.shape[1]
    k = np.array([stats.rankdata(row) for row in full])
    u = (k - (L + 1) / 2).mean(axis=0)
    s = np.sqrt(np.mean(u ** 2))
    z = u[21:24].sum() / (np.sqrt(3) * s)
    assert r.stat == pytest.approx(z)
    # A single event day: Corrado's statistic U_0 / S.
    r0 = inf.rank_test(est, win, slice(2, 3))
    assert r0.stat == pytest.approx(u[22] / s)


def test_rank_test_size_under_null():
    rng = np.random.default_rng(4)
    rej = 0
    for _ in range(600):
        a = rng.standard_t(4, size=(40, 266))
        rej += inf.rank_test(a[:, :250], a[:, 250:], slice(5, 16)).p < 0.05
    assert 0.03 < rej / 600 < 0.075


# ---- bootstrap ------------------------------------------------------------------------------


def test_stationary_bootstrap_block_lengths():
    rng = np.random.default_rng(5)
    idx = bs.stationary_indices(2000, 5.0, 20, rng)
    cont = (np.diff(idx, axis=1) == 1) | ((idx[:, :-1] == 1999) & (idx[:, 1:] == 0))
    # P(continue) = 1 - 1/5 (plus a small chance that a new block starts at the next index).
    assert cont.mean() == pytest.approx(0.8 + 0.2 / 2000, abs=0.01)
    idx1 = bs.stationary_indices(2000, 1.0, 5, rng)
    assert ((np.diff(idx1, axis=1) == 1).mean()) < 0.01  # i.i.d. resampling


def test_block_bootstrap_widens_interval_for_dependent_data():
    rng = np.random.default_rng(6)
    n, phi = 400, 0.7
    e = rng.normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    lo1, hi1 = bs.bootstrap_mean_ci(x, 1.0, 2000, seed=1)
    lo2, hi2 = bs.bootstrap_mean_ci(x, 20.0, 2000, seed=1)
    # Long-run sd of the mean is 1/(1 - phi) times the i.i.d. one, about 3.3 for phi = 0.7.
    ratio = (hi2 - lo2) / (hi1 - lo1)
    assert 1.8 < ratio < 4.0


# ---- multiple testing -----------------------------------------------------------------------


def test_holm_and_bh_match_statsmodels():
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(7)
    for _ in range(20):
        p = np.concatenate([rng.uniform(0, 0.05, 4), rng.uniform(0, 1, 10)])
        np.testing.assert_allclose(mt.holm(p), multipletests(p, method="holm")[1], rtol=1e-12)
        np.testing.assert_allclose(mt.benjamini_hochberg(p),
                                   multipletests(p, method="fdr_bh")[1], rtol=1e-12)
    np.testing.assert_allclose(mt.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


# ---- Sharpe ratios --------------------------------------------------------------------------


def test_dsr_reproduces_the_published_example():
    """Bailey and Lopez de Prado (2014), numerical example: annualised SR 2.5 on 5 years of
    daily data (T = 1250), annualised variance of the trials' SRs 1/2, skewness -3, kurtosis 10.
    The paper reports DSR = 0.9505 after N = 46 trials, a probability of about 90% for its
    N = 100, and that with Normal returns DSR stays above 0.95 up to N = 88."""
    sr = 2.5 / np.sqrt(250)
    v = 0.5 / 250
    assert sh.dsr(sr, 1250, -3.0, 10.0, 46, v) == pytest.approx(0.9505, abs=5e-5)
    assert sh.dsr(sr, 1250, -3.0, 10.0, 100, v) == pytest.approx(0.90, abs=0.005)
    assert sh.dsr(sr, 1250, 0.0, 3.0, 88, v) >= 0.95 > sh.dsr(sr, 1250, 0.0, 3.0, 89, v)


def test_psr_and_annualisation_by_hand():
    # Normal returns, SR* = 0: PSR = Phi(SR sqrt(T - 1) / sqrt(1 + SR^2 / 2)).
    assert sh.psr(0.1, 101, 0.0, 3.0) == pytest.approx(
        stats.norm.cdf(0.1 * 10 / np.sqrt(1 + 0.01 / 2)))
    assert sh.annualise(0.1, 252) == pytest.approx(0.1 * np.sqrt(252))
    assert sh.expected_max_sharpe(1, 1.0) == 0.0
    assert sh.expected_max_sharpe(1000, 1.0) > sh.expected_max_sharpe(10, 1.0) > 0
    x = np.array([0.01, -0.02, 0.03, 0.0, 0.015])
    assert sh.sharpe(x) == pytest.approx(x.mean() / x.std(ddof=1))
    s, k = sh.moments(np.random.default_rng(0).normal(size=200000))
    assert abs(s) < 0.02 and k == pytest.approx(3.0, abs=0.05)


def test_dsr_uses_the_number_of_trials():
    sr, v = 0.15, 0.01
    vals = [sh.dsr(sr, 300, 0.0, 3.0, n, v) for n in (2, 10, 100, 1000)]
    assert all(a > b for a, b in zip(vals, vals[1:], strict=False))


# ---- backtest -------------------------------------------------------------------------------


def _toy_market(n=400, seed=8):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 0.01, n)
    r = np.c_[1.2 * m + rng.normal(0, 0.02, n), 0.5 * m + rng.normal(0, 0.01, n)]
    return r, m


def test_trade_return_by_hand():
    r, m = _toy_market()
    costs = bt.Costs(stock_bps=10, hedge_bps=2, borrow_bps_per_year=50)
    sizing = bt.Sizing(target_vol=0.01, max_weight=3.0)
    tr = bt.trade_returns(r, m, np.array([300, 300]), np.array([0, 1]), np.array([1, -1]), 10,
                          costs, sizing)
    for k, (a, s) in enumerate([(0, 1), (1, -1)]):
        lo = 300 - 280
        x, y = m[lo:270], r[lo:270, a]
        beta, alpha = np.polyfit(x, y, 1)
        sig = np.sqrt(((y - alpha - beta * x) ** 2).sum() / (len(x) - 2))
        w = min(0.01 / sig, 3.0)
        gross = w * s * (r[301:311, a] - beta * m[301:311]).sum()
        cost = 2 * (10 * w + 2 * abs(beta) * w) / 1e4 + (50e-4 * w * 10 / 252 if s < 0 else 0)
        assert tr.loc[k, "beta"] == pytest.approx(beta, rel=1e-9)
        assert tr.loc[k, "gross"] == pytest.approx(gross, rel=1e-9)
        assert tr.loc[k, "net"] == pytest.approx(gross - cost, rel=1e-9)


def test_trade_sizing_ignores_data_after_entry():
    r, m = _toy_market()
    a = bt.trade_returns(r, m, np.array([300]), np.array([0]), np.array([1]), 10)
    r2, m2 = r.copy(), m.copy()
    r2[301:] *= 5
    m2[301:] += 0.05
    b = bt.trade_returns(r2, m2, np.array([300]), np.array([0]), np.array([1]), 10)
    assert a.loc[0, "weight"] == b.loc[0, "weight"] and a.loc[0, "beta"] == b.loc[0, "beta"]


def test_non_overlapping_one_position_per_asset():
    keep = bt.non_overlapping(np.array([10, 12, 15, 21, 22]), np.array([0, 1, 0, 0, 1]), 10)
    np.testing.assert_array_equal(keep, [True, True, False, True, True])


def test_leave_one_season_out_selects_on_other_seasons():
    a = pd.DataFrame({"season": [1, 1, 2, 2, 3], "net": [5.0, 5.0, -1.0, -1.0, 0.0]})
    b = pd.DataFrame({"season": [1, 2, 3], "net": [-10.0, 1.0, 9.0]})
    out = bt.leave_one_season_out({"A": a, "B": b})
    chosen = out.groupby("season")["config"].first().to_dict()
    # Held-out 1: A scores (-1-1+0)/3 = -0.67, B scores (1+9)/2 = 5 -> B.
    # Held-out 2: A (5+5+0)/3 = 3.3, B (-10+9)/2 = -0.5 -> A. Held-out 3: A 2.0, B -4.5 -> A.
    assert chosen == {1: "B", 2: "A", 3: "A"}
    assert out[out.season == 1]["net"].tolist() == [-10.0]


def test_forward_selection_uses_only_earlier_seasons():
    a = pd.DataFrame({"season": [1, 1, 2, 2, 3], "net": [5.0, 5.0, -1.0, -1.0, 0.0]})
    b = pd.DataFrame({"season": [1, 2, 3], "net": [-10.0, 1.0, 9.0]})
    out = bt.forward_selection({"A": a, "B": b}, primary="B")
    chosen = out.groupby("season")["config"].first().to_dict()
    # Season 1: nothing earlier, so the primary B. Season 2: A scores 5, B -10 -> A.
    # Season 3: A scores (5+5-1-1)/4 = 2, B (-10+1)/2 = -4.5 -> A.
    assert chosen == {1: "B", 2: "A", 3: "A"}
    assert out[out.season == 1]["net"].tolist() == [-10.0]
    # No look-ahead: changing the last season's results cannot change an earlier choice, while
    # leave-one-season-out (which also selects on later seasons) does change.
    b2 = b.copy()
    b2.loc[b2.season == 3, "net"] = 1000.0
    out2 = bt.forward_selection({"A": a, "B": b2}, primary="B")
    assert out2.groupby("season")["config"].first().to_dict() == {1: "B", 2: "A", 3: "A"}
    loso_a = bt.leave_one_season_out({"A": a, "B": b}).groupby("season")["config"].first()
    loso_b = bt.leave_one_season_out({"A": a, "B": b2}).groupby("season")["config"].first()
    assert loso_a[2] == "A" and loso_b[2] == "B"  # season 2's choice depends on season 3


def test_event_table_and_metrics_carry_the_gross_return():
    trades = pd.DataFrame({
        "event": ["x", "x", "y", "y", "z"], "season": [1, 1, 2, 2, 3],
        "gross": [0.010, 0.030, -0.020, 0.000, 0.005],
        "cost": [0.002, 0.002, 0.002, 0.002, 0.002],
        "net": [0.008, 0.028, -0.022, -0.002, 0.003]})
    evt = pl.event_table(trades)
    assert evt["event"].tolist() == ["x", "y", "z"]
    np.testing.assert_allclose(evt["gross"], [0.020, -0.010, 0.005])
    np.testing.assert_allclose(evt["net"], evt["gross"] - 0.002)
    m = pl.metrics(evt, n_years=10, seed=1)
    assert m["mean_gross"] == pytest.approx(evt["gross"].mean(), rel=1e-12)
    assert m["mean_gross"] - m["mean_net"] == pytest.approx(0.002, rel=1e-9)
    assert m["gross_lo"] <= m["mean_gross"] <= m["gross_hi"]
