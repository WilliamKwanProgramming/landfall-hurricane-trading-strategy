"""User-supplied price files: schema, validation messages and the alignment with the market."""

import numpy as np
import pandas as pd
import pytest

from landfall import prices as P


def _write(tmp_path, text, name="p.csv"):
    f = tmp_path / name
    f.write_text(text)
    return f


WIDE = """date,TRV,SPX
2020-01-02,100,3000
2020-01-03,101,3030
2020-01-06,,3060
2020-01-07,99.99,3000
"""


def test_wide_file_is_read_with_gaps_as_nan(tmp_path):
    p = P.load_prices(_write(tmp_path, WIDE), market="SPX")
    assert list(p.columns) == ["TRV", "SPX"]
    assert p.index.dtype == "datetime64[s]"
    assert np.isnan(p.loc["2020-01-06", "TRV"]) and p.loc["2020-01-07", "TRV"] == 99.99


def test_long_file_gives_the_same_frame(tmp_path):
    long = ("date,asset,price\n" + "\n".join(
        f"{d},{a},{v}" for d, a, v in [
            ("2020-01-02", "SPX", 3000), ("2020-01-03", "SPX", 3030), ("2020-01-06", "SPX", 3060),
            ("2020-01-07", "SPX", 3000), ("2020-01-02", "TRV", 100), ("2020-01-03", "TRV", 101),
            ("2020-01-07", "TRV", 99.99)]) + "\n")
    a = P.load_prices(_write(tmp_path, long), market="SPX")
    b = P.load_prices(_write(tmp_path, WIDE, "w.csv"), market="SPX")
    pd.testing.assert_frame_equal(a[["TRV", "SPX"]], b, check_freq=False)


@pytest.mark.parametrize("text, message", [
    ("day,TRV,SPX\n2020-01-02,1,2\n", "no 'date' column"),
    ("date,TRV,SPX\n2020-01-02,1,2\n2020-01-02,1,2\n", "duplicate date 2020-01-02"),
    ("date,TRV,SPX\n2020-01-03,1,2\n2020-01-02,1,2\n", "not in increasing order"),
    ("date,TRV,SPX\n02/01/2020,1,2\n03/01/2020,1,2\n", "is not an ISO date"),
    ("date,TRV,SPX\n2020-01-02,1,2\n2020-01-03,abc,2\n", "TRV: 'abc' on 2020-01-03"),
    ("date,TRV,SPX\n2020-01-02,1,2\n2020-01-03,-4,2\n", "not positive and finite"),
    ("date,TRV,SPX\n2020-01-02,1,2\n2020-01-03,0,2\n", "not positive and finite"),
    ("date,TRV,SPX\n2020-01-02,1,2\n2020-01-03,inf,2\n", "not positive and finite"),
    ("date,TRV,IDX\n2020-01-02,1,2\n2020-01-03,1,2\n", "market column 'SPX' not found"),
    ("date,TRV,SPX\n2020-01-02,1,2\n2020-01-03,,2\n", "TRV has fewer than two prices"),
    ("date,asset,price\n2020-01-03,TRV,1\n2020-01-02,TRV,1\n", "for asset TRV"),
])
def test_schema_errors_name_the_problem(tmp_path, text, message):
    with pytest.raises(P.PriceFileError, match=message):
        P.load_prices(_write(tmp_path, text), market="SPX")


def test_alignment_uses_the_common_calendar_for_both_series(tmp_path):
    p = P.load_prices(_write(tmp_path, WIDE), market="SPX")
    a = P.asset_returns(p, "TRV", "SPX")
    # TRV has no price on 2020-01-06, so the last return of both series runs 01-03 -> 01-07.
    assert [d.day for d in a.dates] == [3, 7]
    np.testing.assert_allclose(a.ret, [0.01, 99.99 / 101 - 1], rtol=1e-15)
    np.testing.assert_allclose(a.mkt, [0.01, 3000 / 3030 - 1], rtol=1e-15)


def test_alignment_drops_dates_where_the_market_is_missing():
    idx = pd.DatetimeIndex(pd.date_range("2021-03-01", periods=5, freq="B"))
    p = pd.DataFrame({"OJ": [1.0, 1.1, 1.21, 1.331, 1.4641],
                      "SPX": [10.0, np.nan, 10.0, 11.0, 11.0]}, index=idx)
    a = P.asset_returns(p, "OJ", "SPX")
    assert list(a.dates) == list(idx[[2, 3, 4]])
    np.testing.assert_allclose(a.ret, [0.21, 0.1, 0.1], rtol=1e-13)
    np.testing.assert_allclose(a.mkt, [0.0, 0.1, 0.0], atol=1e-15)


def test_check_prices_reports_gaps_stale_series_and_jumps():
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=60))
    rng = np.random.default_rng(0)
    mkt = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))
    stale = np.repeat(np.linspace(10, 12, 12), 5)  # 48 of 59 returns are zero
    jump = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))
    jump[30:] /= 2.5  # an unadjusted split
    gap = 20 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))
    gap[20:30] = np.nan  # two weeks missing
    p = pd.DataFrame({"M": mkt, "STALE": stale, "JUMP": jump, "GAP": gap}, index=idx)
    rep = P.check_prices(p, "M").set_index("series")
    assert rep.loc["M", "warnings"] == ""
    assert "zero returns" in rep.loc["STALE", "warnings"]
    assert rep.loc["STALE", "zero_return_share"] == pytest.approx(48 / 59, abs=1e-4)
    assert "largest daily move 60%" in rep.loc["JUMP", "warnings"]
    assert rep.loc["GAP", "market_dates_missing"] == 10
    assert rep.loc["GAP", "longest_gap_days"] == 15
    assert "longest gap 15 days" in rep.loc["GAP", "warnings"]
