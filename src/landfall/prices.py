"""Prices supplied by the user (insurers, reinsurers, futures): loading, validation, alignment.

No price data is stored in this repository; this module reads a CSV the user provides on his own
computer (scripts/run_own_prices.py, docs/protocol_own_prices.md).

Schema. Either
- wide: a `date` column (ISO dates, YYYY-MM-DD) and one column per asset holding its daily
  closing price, one row per date, with the market index as one more column; or
- long: columns `date`, `asset`, `price`, one row per (date, asset), the market index being one
  of the assets.
Empty cells are allowed (a series that starts later or has a holiday); every other cell must be
a positive, finite number. Dates must be unique (per asset in the long format) and in increasing
order in the file, so that an export or sorting error is reported instead of silently repaired.

Adjusted versus unadjusted closes matter. For stocks use closes adjusted for splits and
dividends (total-return closes); unadjusted closes put the dividend into the return of the
ex-dividend day as a false negative return, and an unadjusted split looks like a -50% day. For
futures use a continuous series adjusted at the rolls by ratio (multiplicative) adjustment:
unadjusted roll gaps are false returns, and difference (Panama) adjustment can make old prices
negative, which the validation rejects. The market index used in the eight-stock study
(S&P 500) is a price index without dividends; whatever index is used, say which in the protocol
copy.

Alignment. Each asset is aligned with the market separately (`asset_returns`): its calendar is
the set of dates on which both it and the market have a price, and both returns are computed on
that calendar, so a return across a date missing for either series covers the same days for
both. Validation (`check_prices`) reports per asset the share of zero returns (stale prices),
the longest gap between consecutive prices and the market dates without a price.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

MAX_GAP_DAYS = 7
"""Gaps between consecutive prices longer than this many calendar days are reported."""

SPLIT_FACTORS = (1.5, 2.0, 3.0, 4.0, 5.0, 10.0)
"""Common split and reverse-split factors k: a daily gross return within `SPLIT_TOL` (log scale)
of 1/k or k is reported as a possible unadjusted split (a 2:1 split is a return of about -50%,
3:2 about -33%; such days can fall below the 50% move warning)."""

SPLIT_TOL = 0.03

MAX_ZERO_SHARE = 0.10
"""Share of exactly-zero daily returns above which a series is reported as possibly stale (the
event study screens each asset-event with the same 10% limit, docs/protocol.md)."""


class PriceFileError(ValueError):
    """The price file does not follow the documented schema."""


def _parse_dates(raw: pd.Series, path: str) -> pd.DatetimeIndex:
    d = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
    bad = d.isna()
    if bad.any():
        i = int(np.argmax(bad.to_numpy()))
        raise PriceFileError(f"{path}: date {raw.iloc[i]!r} on data row {i + 1} is not an ISO "
                             "date (YYYY-MM-DD)")
    return pd.DatetimeIndex(d).astype("datetime64[s]")


def _check_order(dates: pd.DatetimeIndex, path: str, what: str) -> None:
    v = dates.to_numpy()
    step = np.diff(v)
    dup = np.nonzero(step == np.timedelta64(0, "s"))[0]
    if dup.size:
        raise PriceFileError(f"{path}: duplicate date {pd.Timestamp(v[dup[0]]).date()}{what}")
    back = np.nonzero(step < np.timedelta64(0, "s"))[0]
    if back.size:
        raise PriceFileError(
            f"{path}: dates are not in increasing order{what}: "
            f"{pd.Timestamp(v[back[0]]).date()} is followed by "
            f"{pd.Timestamp(v[back[0] + 1]).date()} (sort the file by date)")


def _to_prices(col: pd.Series, path: str, name: str) -> pd.Series:
    num = pd.to_numeric(col, errors="coerce")
    bad = num.isna() & col.notna() & (col.astype(str).str.strip() != "")
    if bad.any():
        i = int(np.argmax(bad.to_numpy()))
        raise PriceFileError(f"{path}: {name}: {col.iloc[i]!r} on {col.index[i].date()} is not "
                             "a number")
    num = num.astype(float)
    bad = num.notna() & ~(np.isfinite(num) & (num > 0))
    if bad.any():
        i = int(np.argmax(bad.to_numpy()))
        raise PriceFileError(f"{path}: {name}: price {num.iloc[i]} on {num.index[i].date()} is "
                             "not positive and finite (prices must be > 0; see the note on "
                             "adjusted futures series in landfall.prices)")
    return num


def load_prices(path: str | pathlib.Path, market: str) -> pd.DataFrame:
    """Read a wide or long price CSV (schema in the module docstring) and validate it.

    Returns a wide frame indexed by date (datetime64[s]) with one float column per asset, the
    market column included; NaN where a series has no price. Raises PriceFileError with the
    file, column, row or date of the first problem.
    """
    path = str(path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]
    if "date" not in raw.columns:
        raise PriceFileError(f"{path}: no 'date' column (columns: {list(raw.columns)})")
    long = {"asset", "price"} <= set(raw.columns)
    if long:
        extra = set(raw.columns) - {"date", "asset", "price"}
        if extra:
            raise PriceFileError(f"{path}: long format takes only date, asset, price; also "
                                 f"found {sorted(extra)}")
        raw["asset"] = raw["asset"].str.strip()
        frames = {}
        for asset, g in raw.groupby("asset", sort=False):
            dates = _parse_dates(g["date"].reset_index(drop=True), path)
            _check_order(dates, path, f" for asset {asset}")
            frames[asset] = _to_prices(pd.Series(g["price"].to_numpy(), index=dates), path,
                                       str(asset))
        wide = pd.DataFrame(frames).sort_index()
    else:
        dates = _parse_dates(raw["date"], path)
        _check_order(dates, path, "")
        cols = [c for c in raw.columns if c != "date"]
        if len(set(cols)) != len(cols):
            raise PriceFileError(f"{path}: duplicate column names")
        wide = pd.DataFrame({c: _to_prices(pd.Series(raw[c].to_numpy(), index=dates), path, c)
                             for c in cols}, index=dates)
    wide.index.name = "date"
    if market not in wide.columns:
        raise PriceFileError(f"{path}: market column {market!r} not found (assets: "
                             f"{sorted(wide.columns)})")
    if wide.shape[1] < 2:
        raise PriceFileError(f"{path}: needs the market and at least one asset")
    for c in wide.columns:
        if wide[c].notna().sum() < 2:
            raise PriceFileError(f"{path}: {c} has fewer than two prices")
    return wide


@dataclass(frozen=True)
class AssetReturns:
    """Returns of one asset and of the market on their common calendar."""

    asset: str
    dates: pd.DatetimeIndex  # date of each return (the later of the two closes)
    ret: NDArray
    mkt: NDArray


def asset_returns(prices: pd.DataFrame, asset: str, market: str) -> AssetReturns:
    """Simple returns P_t / P_{t-1} - 1 of `asset` and `market` on the dates where both have a
    price; a return spanning a date missing for either series spans it for both."""
    both = prices[[asset, market]].dropna()
    if len(both) < 2:
        raise ValueError(f"{asset} and {market} have fewer than two common dates")
    p = both.to_numpy()
    r = p[1:] / p[:-1] - 1.0
    return AssetReturns(asset, pd.DatetimeIndex(both.index[1:]), r[:, 0], r[:, 1])


def check_prices(prices: pd.DataFrame, market: str, max_gap_days: int = MAX_GAP_DAYS,
                 max_zero_share: float = MAX_ZERO_SHARE) -> pd.DataFrame:
    """One row per series: coverage, gaps, missing market dates, zero returns, largest move.

    `warnings` lists what deserves a look before the run (it does not stop the run: the event
    study screens stale asset-events itself): gaps longer than `max_gap_days` calendar days, a
    share of zero returns above `max_zero_share`, market dates inside the series' range without
    a price, daily moves beyond +-50% (often an unadjusted split or roll), and daily gross
    returns within `SPLIT_TOL` (log) of a split factor 1/k or k, k in `SPLIT_FACTORS`.
    """
    mdates = prices.index[prices[market].notna()]
    rows = []
    for c in prices.columns:
        s = prices[c].dropna()
        gaps = np.diff(s.index.to_numpy()).astype("timedelta64[D]").astype(int)
        r = (s / s.shift(1) - 1.0).iloc[1:]
        inside = mdates[(mdates >= s.index[0]) & (mdates <= s.index[-1])]
        missing = int((~inside.isin(s.index)).sum()) if c != market else 0
        big = float(r.abs().max())
        warn = []
        if gaps.size and gaps.max() > max_gap_days:
            warn.append(f"longest gap {gaps.max()} days")
        zero = float((r == 0).mean())
        if zero > max_zero_share:
            warn.append(f"{zero:.0%} zero returns")
        if missing:
            warn.append(f"{missing} market dates without a price")
        if big > 0.5:
            warn.append(f"largest daily move {big:.0%} on {r.abs().idxmax().date()}")
        lg = np.abs(np.log1p(r.to_numpy()))
        near = np.min(np.abs(lg[:, None] - np.log(SPLIT_FACTORS)[None, :]), axis=1) < SPLIT_TOL \
            if len(lg) else np.zeros(0, dtype=bool)
        if near.any():
            k = int(np.argmax(near))
            warn.append(f"possible unadjusted split: {r.iloc[k]:+.0%} on {r.index[k].date()}")
        rows.append({"series": c, "role": "market" if c == market else "asset",
                     "first_date": s.index[0].date(), "last_date": s.index[-1].date(),
                     "prices": len(s), "longest_gap_days": int(gaps.max()) if gaps.size else 0,
                     "market_dates_missing": missing, "zero_return_share": round(zero, 4),
                     "largest_abs_return": round(big, 4), "warnings": "; ".join(warn)})
    return pd.DataFrame(rows)


def synthetic_prices(names: list[str], market: str, event_times: pd.Series, seed: int,
                     effects: dict[str, float] | None = None,
                     starts: dict[str, str] | None = None, missing_share: float = 0.01,
                     first: str = "1989-01-02", last: str = "2022-12-30") -> pd.DataFrame:
    """SYNTHETIC daily closes, in the schema of `load_prices`, for testing and demonstrating
    scripts/run_own_prices.py. Not market data.

    Market: t(5) returns scaled to 1% daily sd with mean 0.03%. Asset i: beta_i uniform on
    [0.5, 1.3] times the market return plus t(5) noise scaled to 1.5% daily sd, plus, if
    `effects` gives it a total abnormal return c_i, c_i / 11 on each of event days 0 .. +10 of
    every event (day 0 = the first business day whose 20:00 UTC close is at or after the event
    time). A series listed in `starts` begins on that date; `missing_share` of each asset's
    dates are left empty at random (holidays that differ from the market's).
    """
    from landfall.study import eventstudy as es

    effects = effects or {}
    starts = starts or {}
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(pd.bdate_range(first, last)).astype("datetime64[s]")
    n = len(dates)
    scale = np.sqrt(3.0 / 5.0)  # sd of t(5) is sqrt(5/3)
    m = 0.0003 + 0.01 * scale * rng.standard_t(5, n)
    d0 = es.event_rows(dates, event_times)
    d0 = d0[d0 >= 0]
    out = {market: 100.0 * np.cumprod(1.0 + m)}
    for name in names:
        r = rng.uniform(0.5, 1.3) * m + 0.015 * scale * rng.standard_t(5, n)
        c = effects.get(name, 0.0)
        if c:
            for d in d0:
                r[d: min(d + 11, n)] += c / 11.0
        p = 100.0 * np.cumprod(1.0 + r)
        p[rng.random(n) < missing_share] = np.nan
        if name in starts:
            p[dates < pd.Timestamp(starts[name])] = np.nan
        out[name] = p
    df = pd.DataFrame(out, index=dates)
    df.index.name = "date"
    return df[[*names, market]]
