# Method

The formulas and settings behind the README's results. The pre-specified choices of the real-data
study are in `docs/protocol.md`; this file describes what the code does.

**Tracks and landfalls.** HURDAT2 mirrors. Geometric landfall: 10-minute interpolation, first step
on land (1/120-degree mask) after at least 1 h at sea, inside a stated US polygon. A storm has a
"US hurricane landfall" if the interpolated wind at such a step is at least 64 kt; the event of a
storm is its first one. The HURDAT2 "L" flags are the alternative definition.

**Forecast.** For each synoptic fix t, ridge regressions (lambda = 1) fitted on seasons 1950 to
season(t) - 1 predict displacement and intensity change at 12-120 h from position, intensity,
recent motion, day of year and quadratic terms; joint in-sample residual vectors give sampled
tracks; the share entering US land within 120 h is the landfall probability. Verification
(README section 3): forecast origins are fixes with tropical or subtropical status (TD, TS, HU,
SD, SS) that have fixes 12 and 24 h earlier; the fix at which a forecast is verified may have any
status, extratropical stages included; model, persistence and climatology are scored on the same
forecasts; intervals resample whole storms.

**Footprint.** Holland (1980) profile $V(r) = V_m\sqrt{(R_m/r)^B e^{1-(R_m/r)^B}}$, $B$ from the
pressure deficit (clipped to [1, 2.5]), $R_m$ from Willoughby et al. (2006) or the 34-kt radii,
motion asymmetry after Phadke et al. (2003), inland decay $V(t) = V_b + (R V_0 - V_b)e^{-\alpha t}$
(Kaplan and DeMaria 1995). Exposure: expected Emanuel (2011) damage fraction over weighted points.

**Event study.** Market model on days -280..-31, abnormal returns on -5..+10,
$\mathrm{Var}(CAR) = s^2[D + D^2/T + (\sum_w (R_{mt}-\bar R_m))^2/S_{xx}]$; BMP
$t = \overline{SCAR}/(s_{SCAR}/\sqrt N)$, Kolari-Pynnonen factor $\sqrt{(1-\bar r)/(1+(N-1)\bar r)}$,
Corrado-Cowan ranks, stationary bootstrap, placebo dates, Holm and Benjamini-Hochberg. Day 0 is the
first trading day whose close (taken as 20:00 UTC, US daylight time) is at or after the landfall.
Trading: market-hedged, equal ex-ante residual risk, costs per side; the return of an event (one
storm) is the mean over the stocks traded on it; PSR and DSR with
$SR_0 = \sqrt{V}[(1-\gamma)\Phi^{-1}(1-1/N) + \gamma\Phi^{-1}(1-1/(Ne))]$ (reproduces the worked
example of Bailey and Lopez de Prado 2014). The deflated Sharpe ratio picks the best of the 9
configurations by Sharpe ratio and uses N = 9 although the configurations share most trades;
leave-one-season-out and forward-only evaluations pick by mean net return. User-supplied series:
each asset on the dates on which it and the index both have a close.

**Simulated returns and power.** `landfall.study.simulate`: for every event 250 + 30 + 16 days of
a market return and of stock returns $R_{it} = \beta_i R_{mt} + e_{it}$, $\beta_i \sim U(0.5, 1.5)$,
a true abnormal return spread evenly over days 0..+10. Scenarios: Gaussian with constant
volatility, and GARCH(1,1) volatility ($a = 0.08$, $b = 0.90$) with Student-t(5) innovations (market
sd 1% a day; residual sd 2% a day unless set per stock); event-induced variance (residual sd doubled
on days 0..+10) and a shared industry shock ($\rho = 0.3$) are variants. Power is the rejection
rate at 5% over 2,000 replications per cell (`landfall.study.power`); the per-stock power of
README section 1 uses each stock's own number of events and the root mean square of its
market-model residual sds.
