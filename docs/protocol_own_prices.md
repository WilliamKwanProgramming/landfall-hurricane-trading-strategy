# Protocol: insurer, reinsurer and commodity event study on the user's own prices (pre-specified)

Written on 2026-11-05, before any insurer, reinsurer or futures price existed in this repository
or had been looked at by its author. This is the test of the project's central hypothesis that
could not be run where the repository was built. It is run by `scripts/run_own_prices.py` on
prices the person running it supplies, and its output is reported whatever it is. It follows
`docs/protocol.md` (the wave-1 study of eight large US stocks) wherever this file does not say
otherwise; the differences are listed at the end.

## Before loading any price: commit two files

1. Copy this file to `docs/protocol_own_prices_<your initials>.md`, fill in the "Run record"
   section (data source, market index, which listed assets are available, the date), and
   change nothing else. Additions to the asset list are allowed only here, before any price is
   loaded, each with a sign and a reason.
2. Write `assets.csv` (template: `docs/own_prices_assets.csv`) with exactly the assets of your
   copy.
3. Commit both. The commit hash is the time stamp of the pre-registration; the run's MANIFEST
   records the commit, the SHA-256 of the price file and of `assets.csv`.

A run on a list, window, source or setting other than the committed one is a new specification.
Every specification tried, including runs that were discarded, counts towards the deflated
Sharpe ratio: pass their number with `--specifications K` (the script then uses N = 9 K trials)
and list them in the run record.

## Data

- Prices: daily closes supplied by the user in one CSV (schema in `landfall/prices.py`): stocks
  adjusted for splits and dividends; futures as a continuous front-month series with ratio
  adjustment at the rolls. The file stays outside the repository (or in the git-ignored
  `data/`). The loader rejects unsorted or duplicate dates, non-positive and non-finite prices,
  and reports gaps, stale series and daily moves beyond 50%.
- Market index: the S&P 500 (state in the run record whether it is the price or the total-return
  index; wave 1 used the price index).
- Alignment: each asset is aligned with the index on the dates on which both have a close;
  returns are simple close-to-close returns on that common calendar.
- Tracks and events: nothing new is computed from tracks. The run reads three files committed
  with wave 1: `results/real_eventstudy/events.csv` (the event list),
  `results/landfall_counts/us_landfalls_any_intensity_1985_2024.csv` (placebo exclusions) and
  `results/real_eventstudy/landfall_probabilities.csv.gz` (forecast-trigger signals).

## Events

The 39 storms with `kept_primary = True` in `results/real_eventstudy/events.csv`: the first
geometric US hurricane landfall of each storm, 1990-2022, after the overlap rule of
`docs/protocol.md`. The list is not re-derived for the new prices. Day 0 of an asset is the first
date of its calendar whose close (taken as 20:00 UTC) is at or after the landfall time. An
asset-event is used only if its estimation and event windows lie inside that asset's data; the
number not used is reported per asset. The same data screen as wave 1 applies: an asset-event is
excluded if more than 10% of its estimation-window returns or of its event-window returns are
exactly zero.

## Assets and hypotheses

The signs and reasons below are hypotheses written before any price was seen, not findings.

| Group | Asset (template name) | Kind | Sign | Reason (hypothesis) |
|---|---|---|---|---|
| storm-relevant | ALL (Allstate) | stock | - | US home and auto insurer with Gulf and Atlantic coast exposure: expected claims |
| storm-relevant | TRV (Travelers) | stock | - | US property insurer: expected claims |
| storm-relevant | UVE (Universal Insurance Holdings) | stock | - | Florida homeowners insurer: concentrated exposure |
| storm-relevant | HCI (HCI Group) | stock | - | Florida homeowners insurer: concentrated exposure |
| storm-relevant | RNR (RenaissanceRe) | stock | - | property-catastrophe reinsurer: expected claims |
| storm-relevant | EG (Everest Group, ticker RE before 2023) | stock | - | reinsurer with property-catastrophe exposure |
| storm-relevant | OJ (orange-juice futures, ICE FCOJ-A) | future | + | damage to the Florida citrus crop reduces supply |
| storm-relevant | LUMBER (CME lumber futures) | future | + | rebuilding raises lumber demand |
| storm-relevant | NG (NYMEX Henry Hub natural-gas futures) | future | two-sided | Gulf of Mexico production shut-ins raise prices, while power outages, closed LNG export terminals and lower demand lower them; the Gulf's share of US gas output fell over 1990-2022, so the net sign is not specified |
| control | MSFT, JNJ, MRK, PFE | stock | none | no obvious hurricane link (the wave-1 controls) |

An alternative to the insurer hypothesis, stated so that a positive result is not explained
after the fact: after large losses insurers may raise premiums, and investors may expect that.
The protocol still tests the negative sign; a positive CAR is reported as "no evidence for the
hypothesis", with its size.

An asset not available in the user's data is dropped and listed in the run record; it is not
replaced by a similar one after prices have been loaded.

## Abnormal returns and tests

As in `docs/protocol.md`: market model by OLS on trading days -280 .. -31 of each asset's
calendar; event window -5 .. +10; CAR windows [-5, -1], [0, +10], [-5, +10]. Per asset: mean CAR,
BMP t (primary), Patell Z, naive t, Corrado-Cowan rank test, 95% stationary-bootstrap interval
(2,000 resamples, mean block 2 events, seed 20261105). For futures the S&P 500 market model is
kept for comparability; with a beta near 0 the abnormal return is close to the return minus its
estimation-window mean.

Pooled: the signed CAR (sign times CAR) of the storm-relevant assets with a one-sided sign, BMP t
with the Kolari-Pynnonen adjustment and a stationary-bootstrap interval (mean block 4); the
controls pooled unsigned. Two-sided assets are tested individually only.

Multiple testing family: the BMP p-values of every asset in `assets.csv` (storm-relevant and
controls) for the three windows (with the full template list: 13 assets x 3 windows = 39),
adjusted with Holm (family-wise error 5%) and Benjamini-Hochberg (false discovery rate 5%).

Placebo: 1,000 placebo sets drawn as in `docs/protocol.md` (same calendar week of a different
year in 1991-2022, at least 21 days from any US landfall of any intensity), the dates drawn on the
market index's calendar and mapped to each asset's calendar; placebo asset-events go through the
same data screen. Reported per asset and for the pooled signed CAR[0, +10]: the fraction of
placebo values at or below the real value and the two-sided placebo p-value. Seed 20261105.

## What counts as a finding

- Per asset: CAR[0, +10] with Holm-adjusted BMP p < 0.05, placebo two-sided p < 0.05, and the
  hypothesised sign (for natural gas: either sign). Anything else is reported as no evidence.
- Pooled: signed pooled CAR[0, +10] with BMP-KP p < 0.05 and placebo two-sided p < 0.05.
- A control meeting the per-asset criterion is reported as a warning that the design produces
  false positives.

## Statistical power (state it with the result)

With about 35 events per asset the wave-1 simulation under GARCH-t returns
(`results/synthetic_eventstudy/garch_t_35x1.csv`, BMP test) detects a CAR of 1% in about 16% of
samples and of 2% in about 44%. Insurer and futures returns are more volatile than the wave-1 stocks, so the
detectable effect is larger still. A null result per asset therefore rules out only large
effects; the report states the bootstrap interval of each mean CAR so that the reader sees which
effects are excluded.

## Trading rules

As in `docs/protocol.md` (R1 calendar rule, not tradeable; R2 forecast-trigger rule, tradeable;
the same 9 configurations each, the same primaries (-5, +10) and (0.4, 10), the same sizing and
leave-one-season-out evaluation), traded on the storm-relevant assets with a one-sided sign (the
natural-gas position has no pre-specified direction and is not traded). Costs (assumptions):
10 bp per side on stocks and on futures, 2 bp per side on the index hedge, 50 bp per year borrow
on short stock positions, none on short futures. Futures returns are price returns of the
continuous series (no collateral interest).

Deflated Sharpe ratio of the best configuration of each rule with N = 9 K trials, K the number
of specifications run (see above), and the variance of the 9 configurations' per-event Sharpe
ratios. Finding for R2: leave-one-season-out mean event return positive with a bootstrap interval
that excludes 0, and a deflated Sharpe ratio above 0.95.

## Sensitivities (reported, not used for findings)

1. All 48 events kept (no overlap rule).
2. Costs doubled for R1 and R2.

## Differences from docs/protocol.md

- Assets, kinds and signs as above; a two-sided asset (natural gas) is tested individually but
  not pooled or traded.
- Each asset has its own calendar (dates on which it and the index both have a close).
- The event list, the placebo exclusions and the forecast probabilities are read from the wave-1
  results instead of being recomputed.
- The flag-based event-list sensitivity is not repeated (it needs the track files).
- The deflated Sharpe ratio counts N = 9 K trials.

## Limitations known in advance

- Day 0 assumes a 20:00 UTC close for every series; orange juice, natural gas and lumber
  futures settle earlier in the US afternoon, so a landfall between the settlement and 20:00
  UTC is assigned to that day's close although the futures had already settled.
- Continuous front-month futures series mix contracts; roll dates inside event windows add noise.
- The forecast trigger comes from the statistical model of this repository, not from NHC
  advisories; the result says nothing about NHC forecasts as issued.

## Run record (filled in by the person running it, in his copy, before loading prices)

- Name, date:
- Price source and download date:
- Market index (price or total return):
- Assets available / dropped / added (with sign and reason):
- Number of specifications K run so far (including this one):
- Commit of this copy and of assets.csv:

## Deviations

- 2026-11-05, before any real price was used: `experiments/placebo_size.py` showed that the placebo
  test rejects a true null more often than its level on synthetic prices without any effect (15.8%
  at 5%, `results/placebo_size/summary.csv`), while BMP holds its level. The criterion above is not
  changed, because it also requires the Holm-adjusted BMP p-value; a small placebo p-value on its
  own is to be reported as weak evidence.
