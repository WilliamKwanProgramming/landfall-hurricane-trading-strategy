"""ATCF a-deck reader on a fixture written from the NRL ATCF format documentation."""

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES

from landfall.data import atcf


@pytest.fixture(scope="module")
def adeck():
    return atcf.parse_adeck((FIXTURES / "adeck_sample.dat").read_text())


def test_one_row_per_forecast_point(adeck):
    # 11 lines; the three 24-h lines of the first forecast (34/50/64 kt radii) merge into one.
    assert len(adeck) == 9
    assert set(adeck["tech"]) == {"OFCL", "CLP5"}
    assert sorted(adeck["storm_id"].unique()) == ["AL982022", "AL992011"]


def test_units_and_hemispheres(adeck):
    r = adeck[(adeck.storm_id == "AL992011") & (adeck.tech == "OFCL") & (adeck.tau_h == 0)].iloc[0]
    assert r["lat"] == pytest.approx(25.4) and r["lon"] == pytest.approx(-76.8)
    assert r["issue_time"] == pd.Timestamp("2011-08-24 00:00")
    assert r["vmax_kt"] == 50 and r["mslp_mb"] == 990 and r["rmw_nm"] == 25


def test_merged_wind_radii(adeck):
    r = adeck[(adeck.storm_id == "AL992011") & (adeck.tau_h == 24) & (adeck.tech == "OFCL")
              & (adeck.issue_time == pd.Timestamp("2011-08-24 00:00"))].iloc[0]
    assert [r["r34_ne"], r["r34_se"], r["r34_sw"], r["r34_nw"]] == [120, 90, 40, 70]
    assert [r["r50_ne"], r["r50_nw"], r["r64_ne"], r["r64_se"]] == [40, 20, 20, 0]
    assert np.isnan(r["mslp_mb"])  # MSLP 0 = not forecast


def test_short_line_and_errors(adeck):
    clp = adeck[adeck.tech == "CLP5"].iloc[0]
    assert clp["lat"] == pytest.approx(26.8) and np.isnan(clp["r34_ne"])
    with pytest.raises(ValueError):
        atcf.parse_adeck("AL, 99, 2011082400, 03, OFCL,   0, 254N")
    with pytest.raises(ValueError):
        atcf.parse_adeck("AL, 99, 2011082400, 03, OFCL,   0, 254N,  768Q,  50,")


def test_availability_rule_uses_delay(adeck):
    f = atcf.official(adeck)
    t = pd.Timestamp("2011-08-24 08:59")
    fc = atcf.available_forecast(f, "AL992011", t, delay_hours=3.0)
    assert fc.issue_time == pd.Timestamp("2011-08-24 00:00")  # the 06 UTC one is public at 09
    fc = atcf.available_forecast(f, "AL992011", t + pd.Timedelta(minutes=1), delay_hours=3.0)
    assert fc.issue_time == pd.Timestamp("2011-08-24 06:00")
    np.testing.assert_array_equal(fc.leads_h, [0, 12, 24])
    assert atcf.available_forecast(f, "AL992011", pd.Timestamp("2011-08-24 02:00")) is None
    # Zero delay: public at the synoptic time itself.
    fc0 = atcf.available_forecast(f, "AL992011", pd.Timestamp("2011-08-24 06:00"), 0.0)
    assert fc0.issue_time == pd.Timestamp("2011-08-24 06:00")


def test_later_lines_cannot_change_an_available_forecast(adeck):
    f = atcf.official(adeck)
    t = pd.Timestamp("2011-08-24 04:00")
    before = atcf.available_forecast(f, "AL992011", t)
    later = f[f.issue_time > t - pd.Timedelta(hours=3)]
    altered = pd.concat([f[f.issue_time <= t - pd.Timedelta(hours=3)],
                         later.assign(lat=later.lat + 5.0)])
    after = atcf.available_forecast(altered, "AL992011", t)
    np.testing.assert_array_equal(before.lat, after.lat)
    assert before.issue_time == after.issue_time
