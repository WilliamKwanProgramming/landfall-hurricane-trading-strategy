"""Daily price loaders. No price data is stored in this repository.

`load_skfolio_prices` reads the datasets bundled in the PyPI wheel of skfolio (installed with
`pip install skfolio`): adjusted closes of 20 US stocks and the S&P 500 index, 1990-01-02 to
2022-12-28. skfolio documents both as "daily adjusted closing prices". Checked here: splits are
adjusted (no jump at the AAPL splits of 2014-06-09 and 2020-08-31), and the index equals the
closing level of the price index in arch.data.sp500 (largest difference 0.0001 over 5,031 common
days), i.e. the index excludes dividends. Whether the stock series also include dividends (as
Yahoo's "Adj Close" does) could not be checked against an independent source in the sandbox
where this was written. RRC (Range Resources) has many unchanged closes before 1996 (about 55% of
days in 1990-1995) and a -67% close-to-close change on 1990-04-10; the event study screens for
stale prices (see docs/protocol.md).

Prices a user supplies (insurers, reinsurers, futures) are read by `landfall.prices`.
"""

from __future__ import annotations

import pandas as pd


def load_skfolio_prices() -> tuple[pd.DataFrame, pd.Series]:
    """(stock prices, S&P 500 index level) indexed by trading date."""
    from skfolio.datasets import load_sp500_dataset, load_sp500_index

    p = load_sp500_dataset()
    i = load_sp500_index()["SP500"]
    p.index = pd.DatetimeIndex(p.index).astype("datetime64[s]")
    i.index = pd.DatetimeIndex(i.index).astype("datetime64[s]")
    return p, i


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Daily simple returns P_t / P_{t-1} - 1; the first row is dropped."""
    return (prices / prices.shift(1) - 1.0).iloc[1:]
