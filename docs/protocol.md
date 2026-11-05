# Protocol: real-data event study with the available stocks (pre-specified)

Written and committed before any return around any landfall was computed. The study is run once
by `experiments/real_eventstudy.py`, and its output is reported whatever it is. Any later change
is listed under "Deviations" at the end, with the date and the reason. Before writing this
protocol the price data were inspected only for quality (largest daily changes, unchanged
closes, agreement of the index with another source); no event-window or estimation-window
return was looked at.

## What this study can and cannot show

The project's question is whether prices of hurricane-exposed securities (insurers, reinsurers,
orange juice, natural gas, lumber) fully reflect National Hurricane Center forecasts. Those prices
are not available where this repository was built. This study uses the only daily prices that
are: 20 large US stocks and the S&P 500 index (skfolio datasets, 1990-01-02 to 2022-12-28). It
asks a narrower question: do four stocks with a plausible hurricane link show abnormal returns
around US hurricane landfalls, compared with four stocks without an obvious link? A null result
here says nothing about insurers or commodities.

## Data

- Prices: `skfolio.datasets.load_sp500_dataset()` (adjusted closes) and `load_sp500_index()`
  (S&P 500 price index, no dividends); simple close-to-close returns.
- Tracks: HURDAT2 rda mirror for 1851-2019 and dplyr mirror for 2020-2024
  (`landfall.data.tracks.load_combined_tracks`).

## Events

- Storms with at least one geometric landfall (`landfall.events.geometric_landfalls`: 10-minute
  interpolation of the synoptic track, at least 1 h at sea before the crossing, land mask of
  `global-land-mask`) inside the US coast polygon (`landfall.events.US_COAST_POLYGON`) with
  interpolated maximum wind >= 64 kt. One event per storm, at its first such landfall. Seasons
  1990-2022. The list is `results/landfall_counts/us_hurricane_landfalls_1990_2022.csv`
  (48 storms), produced by `experiments/landfall_counts.py` before this protocol.
- Day 0: the first trading day whose close (taken as 20:00 UTC, 16:00 US Eastern daylight time)
  is at or after the landfall time.
- An event is used only if its estimation and event windows lie inside the price data.
- Overlapping events (primary rule): events sorted by day 0; an event is dropped if its day 0 is
  15 trading days or fewer after the day 0 of the last kept event (their [-5, +10] windows would
  overlap). The number dropped is reported. Sensitivity: all events kept.

## Assets and hypotheses

| Group | Ticker | Pre-specified sign | Reason (hypothesis, not a finding) |
|---|---|---|---|
| storm-relevant | HD | + | rebuilding demand for home-improvement goods |
| storm-relevant | XOM | - | damage to and shut-ins of Gulf production and refining |
| storm-relevant | CVX | - | same as XOM |
| storm-relevant | RRC | + | natural-gas producer; Gulf shut-ins raise gas prices. Its geographic exposure over 1990-2022 was not checked here |
| control | MSFT, JNJ, MRK, PFE | none | no obvious hurricane link |

Market: the S&P 500 index.

Data-quality screen: an asset-event is excluded if more than 10% of the daily returns in its
estimation window, or more than 10% in its event window, are exactly zero (stale prices; RRC has
many unchanged closes before 1996). Excluded asset-events are counted in the output.

## Abnormal returns

Market model by OLS on the 250 trading days -280 .. -31 (ending 30 trading days before day 0;
no overlap with the event window). Event window -5 .. +10. CAR windows: [-5, -1], [0, +10],
[-5, +10]. Formulas in `landfall/study/eventstudy.py`.

## Tests (per asset, then pooled)

For each asset and window: mean CAR; BMP standardised cross-sectional t (primary); Patell Z;
naive cross-sectional t; Corrado-Cowan rank test; 95% stationary-bootstrap interval of the mean
CAR (2,000 resamples, mean block 2 events, seed 20261105).

Pooled storm-relevant group: signed CAR (sign times CAR) of the four storm-relevant stocks, BMP t
with the Kolari-Pynnonen adjustment (stocks of one event share dates), and a stationary-bootstrap
interval with mean block 4. Pooled controls: unsigned CAR, same tests.

Multiple testing: the 24 BMP p-values (8 assets x 3 windows) adjusted with Holm (family-wise
error 5%) and Benjamini-Hochberg (false discovery rate 5%).

Placebo: 1,000 placebo sets. For each event, a placebo date is drawn uniformly from the seven
days of the same ISO calendar week in a different year drawn uniformly from 1991-2022, and
redrawn if a US landfall of any intensity (geometric definition, US polygon) lies within 21
calendar days of it or if its windows fall outside the price data. On each placebo set: per-asset
mean CAR[0, +10] and the pooled signed mean CAR[0, +10]. Reported: where the real value lies in
the placebo distribution (fraction of placebo values at or below it) and the two-sided placebo
p-value 2 min(fraction <=, fraction >=). Seed 20261105.

## What counts as a finding

- Per asset: CAR[0, +10] of a storm-relevant stock with Holm-adjusted BMP p < 0.05, placebo
  two-sided p < 0.05, and the pre-specified sign. Any other outcome is reported as no evidence.
- Pooled: signed pooled CAR[0, +10] with BMP-KP p < 0.05 and placebo two-sided p < 0.05.
- A control stock meeting the per-asset criterion is reported as a warning that the design
  produces false positives.

## Trading rules

Positions are market-hedged and sized for equal risk (`landfall/study/backtest.py`): weight
w = min(1% / residual sd, 3), beta and residual sd from the 250 days ending 30 days before entry.
Costs (assumptions): 10 bp per side on the stock, 2 bp per side on the index hedge (futures, no
borrow), 50 bp per year borrow on short stock. Signs as in the table; storm-relevant stocks only.
The return of an event is the mean net return over the stocks traded on it.

- R1, calendar rule (uses the landfall date, so it could not have been traded in real time; a
  benchmark only): enter at the close of day -5, exit at the close of day +10. Configurations
  tried: entry day in {-5, -3, -1} x exit day in {+3, +5, +10} (9). Primary: (-5, +10).
- R2, forecast-trigger rule (tradeable): for every storm of 1990-2022, landfalling or not, the
  point-in-time model (`landfall.signals`, trained on 1950 to the previous season, 200 sampled
  tracks per forecast, seed 20261105, forecast public 3 h after its synoptic time) gives the
  probability of a US landfall within 120 h. Enter at the first close at or after the public
  time of the first forecast with probability >= p*; exit H trading days later; one open position
  per stock at a time. Configurations: p* in {0.2, 0.4, 0.6} x H in {5, 10, 15} (9).
  Primary: (0.4, 10).

Metrics per configuration: number of events, mean event return with a 95% stationary-bootstrap
interval (2,000 resamples, mean block 2), Sharpe ratio per event and annualised with the number
of events per year, probabilistic Sharpe ratio against 0. Deflated Sharpe ratio of the best
configuration of each rule, with N = 9 (the configurations tried) and the variance of their
per-event Sharpe ratios. Leave-one-season-out: for each season, the configuration with the
highest mean event return on the other seasons is applied to that season; the held-out event
returns are pooled and reported with their bootstrap interval.

Finding for R2: the leave-one-season-out mean event return is positive with a bootstrap interval
that excludes 0, and the deflated Sharpe ratio of the best configuration exceeds 0.95.

## Sensitivities (reported, not used for findings)

1. All events kept (no overlap rule).
2. Events from the HURDAT2 landfall flags (rda mirror), 1990-2019.
3. Costs doubled for R1 and R2.

## Deviations

None so far.
