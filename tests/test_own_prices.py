"""End-to-end test of scripts/run_own_prices.py on SYNTHETIC prices generated here.

The prices are simulated (landfall.prices.synthetic_prices) and written to pytest's temporary
directory; every output is labelled "synthetic". Two insurers get an injected abnormal return
of -6% over days 0..+10 of every event, so the run must find them (and only them) and the
calendar rule, which shorts them, must earn money before costs.
"""

import hashlib
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

from landfall import pipeline as pl
from landfall.prices import synthetic_prices

REPO = pathlib.Path(__file__).resolve().parents[1]

ASSETS = """asset,group,kind,sign,reason
TRV,storm,stock,-,insurer (test)
ALL,storm,stock,-,insurer (test)
NG,storm,future,two-sided,gas (test)
OJ,storm,future,+,orange juice (test)
MSFT,control,stock,none,control (test)
"""


def _script():
    spec = importlib.util.spec_from_file_location("run_own_prices",
                                                  REPO / "scripts" / "run_own_prices.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("own")
    events = pd.read_csv(REPO / "results" / "real_eventstudy" / "events.csv",
                         parse_dates=["time"])
    df = synthetic_prices(["TRV", "ALL", "NG", "OJ", "MSFT"], "SPX", events["time"], seed=7,
                          effects={"TRV": -0.06, "ALL": -0.06}, starts={"OJ": "2008-07-01"})
    prices = tmp / "prices.csv"
    df.to_csv(prices, float_format="%.6f", date_format="%Y-%m-%d")
    assets = tmp / "assets.csv"
    assets.write_text(ASSETS)
    out = tmp / "out"
    rc = _script().main(["--prices", str(prices), "--assets", str(assets), "--market", "SPX",
                         "--label", "synthetic", "--out", str(out), "--placebo-sets", "20",
                         "--power-reps", "10", "--specifications", "2"])
    assert rc == 0
    return out, prices, assets, df


def test_every_output_is_labelled_synthetic(run):
    out = run[0]
    csvs = sorted(out.glob("*.csv"))
    assert len(csvs) == 15
    assert (out / "trading_forward.csv").exists() and (out / "power_by_asset.csv").exists()
    for f in csvs:
        d = pd.read_csv(f)
        assert list(d.columns)[0] == "data" and (d["data"] == "synthetic").all(), f.name
    man = (out / "MANIFEST.txt").read_text()
    assert "data: SYNTHETIC" in man


def test_manifest_has_the_input_checksums_and_no_price_is_written(run):
    out, prices, assets, df = run
    man = (out / "MANIFEST.txt").read_text()
    assert f"prices_sha256: {hashlib.sha256(prices.read_bytes()).hexdigest()}" in man
    assert f"assets_sha256: {hashlib.sha256(assets.read_bytes()).hexdigest()}" in man
    assert "dsr_trials: 18" in man
    text = "".join(f.read_text() for f in out.glob("*.csv"))
    sample = df["TRV"].dropna().iloc[[100, 2000, 5000]]
    assert not any(f"{v:.6f}" in text or f"{v:.6g}" in text for v in sample)


def test_power_is_reported_at_each_assets_own_events_and_volatility(run):
    out = run[0]
    pw = pd.read_csv(out / "power_by_asset.csv")
    cov = pd.read_csv(out / "coverage.csv").set_index("asset")
    assert list(pw["asset"]) == list(cov.index)
    assert (pw["events"].to_numpy() == cov.loc[pw["asset"], "used"].to_numpy()).all()
    assert (pw["resid_sd_rms"] > 0).all()
    assert ((pw["size_bmp"] >= 0) & (pw["size_bmp"] <= 1)).all()
    assert "power_reps: 10" in (out / "MANIFEST.txt").read_text()


def test_injected_effects_are_found_and_nothing_else(run):
    out = run[0]
    f = pd.read_csv(out / "findings_event_study.csv").set_index("series")
    assert f.loc["TRV", "meets_criterion"] and f.loc["ALL", "meets_criterion"]
    assert not f.loc["MSFT", "meets_criterion"] and not f.loc["NG", "meets_criterion"]
    assert not f.loc["OJ", "meets_criterion"]
    assert f.loc["storm (signed, pooled)", "meets_criterion"]
    car = pd.read_csv(out / "car_tests.csv")
    post = car[car.window == "post[0,+10]"].set_index("asset")
    assert post.loc["TRV", "mean_car"] == pytest.approx(-0.06, abs=0.02)
    assert post.loc["MSFT", "boot_lo"] < 0 < post.loc["MSFT", "boot_hi"]


def test_two_sided_asset_is_tested_but_not_pooled_or_traded(run):
    out = run[0]
    car = pd.read_csv(out / "car_tests.csv")
    assert "NG" in set(car["asset"])
    pooled = pd.read_csv(out / "pooled.csv")
    storm = pooled[(pooled.group == "storm (signed)") & (pooled.window == "post[0,+10]")]
    cov = pd.read_csv(out / "coverage.csv").set_index("asset")
    assert storm["n"].item() == cov.loc[["TRV", "ALL", "OJ"], "used"].sum()


def test_short_history_and_calendar_rule_sign(run):
    out = run[0]
    cov = pd.read_csv(out / "coverage.csv").set_index("asset")
    assert cov.loc["OJ", "outside_data"] > 10 and cov.loc["TRV", "outside_data"] == 0
    cfg = pd.read_csv(out / "trading_configs.csv")
    r1 = cfg[(cfg.rule.str.startswith("R1")) & cfg.primary].iloc[0]
    assert r1["mean_net"] > 0.01  # shorting insurers that fall 6% earns money


def test_assets_file_is_validated(tmp_path):
    bad = tmp_path / "a.csv"
    for text, msg in [
        ("asset,group,kind,sign,reason\nX,storm,stock,negative,r\n", "sign of a storm"),
        ("asset,group,kind,sign,reason\nX,control,stock,-,r\n", "sign of a control"),
        ("asset,group,kind,sign,reason\nX,storm,bond,-,r\n", "kind must be"),
        ("asset,group,kind,sign,reason\nX,storm,stock,-,\n", "reason"),
        ("asset,group,kind,sign\nX,storm,stock,-\n", "columns"),
        ("asset,group,kind,sign,reason\nX,storm,stock,-,r\nX,storm,stock,+,r\n", "duplicate"),
    ]:
        bad.write_text(text)
        with pytest.raises(ValueError, match=msg):
            pl.load_assets(bad)
    good = pl.load_assets(REPO / "docs" / "own_prices_assets.csv")
    signs = {a.name: a.sign for a in good}
    assert signs["TRV"] == -1 and signs["OJ"] == 1 and signs["NG"] == 0 and signs["MSFT"] == 0
    assert [a.directional for a in good].count(True) == 8


def test_missing_asset_column_is_an_error(tmp_path):
    idx = pd.bdate_range("2000-01-03", periods=10)
    pd.DataFrame({"TRV": np.linspace(10, 11, 10), "SPX": np.linspace(1, 2, 10)}, index=idx) \
        .rename_axis("date").to_csv(tmp_path / "p.csv", date_format="%Y-%m-%d")
    (tmp_path / "a.csv").write_text(ASSETS)
    with pytest.raises(SystemExit, match="no column for"):
        _script().main(["--prices", str(tmp_path / "p.csv"), "--assets",
                        str(tmp_path / "a.csv"), "--label", "synthetic", "--out",
                        str(tmp_path / "o")])


def test_placebo_on_an_own_calendar_equals_the_event_study_at_the_placebo_dates(run):
    """The placebo maps dates drawn on the market calendar to each asset's calendar; the
    result must equal an event study run directly at those dates."""
    from landfall import prices as P
    from landfall import realstudy as rs

    _, prices_path, assets_path, _ = run
    prices = P.load_prices(prices_path, "SPX")
    datas = []
    for a in pl.load_assets(assets_path):
        r = P.asset_returns(prices, a.name, "SPX")
        datas.append(pl.AssetData(a, r.dates, r.ret, r.mkt))
    events = pd.read_csv(REPO / "results" / "real_eventstudy" / "events.csv",
                         parse_dates=["time"]).head(8)
    excl = pd.read_csv(REPO / "results" / "landfall_counts" /
                       "us_landfalls_any_intensity_1985_2024.csv", parse_dates=["time"])
    cal = pd.DatetimeIndex(prices["SPX"].dropna().index[1:])
    got, _ = pl.placebo(datas, cal, events["time"], excl["time"], (1991, 2022), 3, 5)
    p0 = rs.placebo_day0(events["time"], excl["time"], cal, (1991, 2022), 3,
                         np.random.default_rng([5, 1]))
    for d in datas:
        for s in range(3):
            fits, _ = pl.fit_assets([d], pd.Series(cal[p0[s]]) + pd.Timedelta(hours=12))
            f = fits[d.asset.name]
            want = f.mm.car(0, 10).mean() if f is not None else np.nan
            np.testing.assert_allclose(got[d.asset.name][s], want, rtol=1e-12)
