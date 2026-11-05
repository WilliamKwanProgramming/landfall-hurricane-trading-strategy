"""The event-study and trading pipeline of docs/protocol.md, for any list of assets.

Used by experiments/real_eventstudy.py (wave 1: eight stocks on one calendar) and by
scripts/run_own_prices.py (docs/protocol_own_prices.md: the user's insurer, reinsurer and futures
prices, each asset on its own calendar). Every asset carries its own dates, returns and the
market's returns on those dates (`AssetData`); events are given as times and mapped to each
asset's calendar, so assets with different histories or holidays need no common calendar.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from landfall import realstudy as rs
from landfall.study import backtest as bt
from landfall.study import bootstrap as bs
from landfall.study import eventstudy as es
from landfall.study import inference as inf
from landfall.study import multiple as mt
from landfall.study import sharpe as sh

SIGNS = {"+": 1, "-": -1, "two-sided": 0, "none": 0}
GROUPS = ("storm", "control")
KINDS = ("stock", "future")


@dataclass(frozen=True)
class Asset:
    """An asset of the protocol: group 'storm' or 'control', sign +1 / -1 (hypothesised
    direction) or 0 (two-sided for a storm-relevant asset; none for a control)."""

    name: str
    group: str
    sign: int
    kind: str = "stock"

    @property
    def directional(self) -> bool:
        """Storm-relevant with a one-sided hypothesis: pooled (signed) and traded."""
        return self.group == "storm" and self.sign != 0


@dataclass(frozen=True)
class AssetData:
    asset: Asset
    dates: pd.DatetimeIndex
    ret: NDArray
    mkt: NDArray


def load_assets(path: str | pathlib.Path) -> list[Asset]:
    """Read assets.csv (columns asset, group, kind, sign, reason; docs/own_prices_assets.csv).

    sign is '+', '-' or 'two-sided' for group 'storm' and 'none' for group 'control'.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = {"asset", "group", "kind", "sign", "reason"}
    if not need <= set(df.columns):
        raise ValueError(f"{path}: columns {sorted(need)} required, found {list(df.columns)}")
    out = []
    for i, r in df.iterrows():
        where = f"{path}, row {i + 1} ({r['asset']})"
        name, group, kind, sign = (r[k].strip() for k in ("asset", "group", "kind", "sign"))
        if not name:
            raise ValueError(f"{where}: empty asset name")
        if group not in GROUPS:
            raise ValueError(f"{where}: group must be one of {GROUPS}, not {group!r}")
        if kind not in KINDS:
            raise ValueError(f"{where}: kind must be one of {KINDS}, not {kind!r}")
        allowed = ("none",) if group == "control" else ("+", "-", "two-sided")
        if sign not in allowed:
            raise ValueError(f"{where}: sign of a {group} asset must be one of {allowed}, "
                             f"not {sign!r}")
        if not r["reason"].strip():
            raise ValueError(f"{where}: the reason (hypothesis) is empty")
        out.append(Asset(name, group, SIGNS[sign], kind))
    names = [a.name for a in out]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate asset names")
    return out


def day0_rows(data: AssetData, times: pd.Series) -> NDArray:
    """Day-0 row of each event on the asset's calendar; -1 if the windows do not fit."""
    d0 = es.event_rows(data.dates, times)
    fits = np.array([rs.window_fits(int(d), len(data.dates)) for d in d0], dtype=bool)
    return np.where(fits, d0, -1)


@dataclass
class Fit:
    data: AssetData
    events: NDArray  # indices into the event list
    mm: es.MarketModel


def fit_assets(datas: list[AssetData], times: pd.Series) -> tuple[dict, pd.DataFrame]:
    """Market models of every asset for the events at `times`, and a coverage table
    (events outside the asset's data, excluded by the stale-price screen, used)."""
    fits, cov = {}, []
    for d in datas:
        d0 = day0_rows(d, times)
        inside = np.nonzero(d0 >= 0)[0]
        fit, screened = (None, 0) if inside.size == 0 else rs.fit_asset(
            d.ret, d.mkt, d0[inside], d.asset.name)
        fits[d.asset.name] = None if fit is None else Fit(d, inside[fit.events], fit.mm)
        cov.append({"asset": d.asset.name, "events": len(times),
                    "outside_data": int(len(times) - inside.size), "screened_out": screened,
                    "used": 0 if fit is None else len(fit.events)})
    return fits, pd.DataFrame(cov)


def event_study(datas: list[AssetData], times: pd.Series, seed: int
                ) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Per-asset tests, pooled tests (signed directional storm assets; controls), the fits and
    the coverage table."""
    fits, cov = fit_assets(datas, times)
    rows, pooled = [], []
    for d in datas:
        a = d.asset
        fit = fits[a.name]
        screened = int(cov.loc[cov.asset == a.name, "screened_out"].iloc[0])
        for wname, (t1, t2) in es.CAR_WINDOWS.items():
            r = {"asset": a.name, "group": a.group, "window": wname, "screened_out": screened}
            if fit is not None:
                r.update(rs.window_stats(fit.mm.car(t1, t2), fit.mm.scar(t1, t2), fit.mm, t1, t2,
                                         2.0, seed))
            rows.append(r)
    groups = (("storm (signed)", [d.asset for d in datas if d.asset.directional], True),
              ("control", [d.asset for d in datas if d.asset.group == "control"], False))
    for group, assets, signed in groups:
        parts = [(a, fits[a.name]) for a in assets if fits[a.name] is not None]
        if not parts:
            continue
        for wname, (t1, t2) in es.CAR_WINDOWS.items():
            sg = [a.sign if signed else 1 for a, _ in parts]
            car = np.concatenate([s * f.mm.car(t1, t2) for s, (_, f) in zip(sg, parts,
                                                                              strict=True)])
            scar = np.concatenate([s * f.mm.scar(t1, t2) for s, (_, f) in zip(sg, parts,
                                                                                strict=True)])
            ev_id = np.concatenate([f.events for _, f in parts])
            resid = np.vstack([s * f.mm.resid_est for s, (_, f) in zip(sg, parts, strict=True)])
            ar = np.vstack([s * f.mm.ar for s, (_, f) in zip(sg, parts, strict=True)])
            order = np.argsort(ev_id, kind="stable")
            car, scar, ev_id, resid, ar = (x[order] for x in (car, scar, ev_id, resid, ar))
            rbar = inf.average_cross_correlation(resid, ev_id)
            lo, hi = bs.bootstrap_mean_ci(car, 4.0, 2000, seed)
            cols = slice(t1 - es.WINDOW[0], t2 - es.WINDOW[0] + 1)
            kp = inf.bmp_kp(scar, rbar)
            pooled.append({"group": group, "window": wname, "n": len(car),
                           "mean_car": car.mean(), "boot_lo": lo, "boot_hi": hi,
                           "bmp_t": inf.bmp(scar).stat, "bmp_p": inf.bmp(scar).p,
                           "rbar": rbar, "bmp_kp_t": kp.stat, "bmp_kp_p": kp.p,
                           "rank_z": inf.rank_test(resid, ar, cols).stat,
                           "rank_p": inf.rank_test(resid, ar, cols).p})
    per_asset = pd.DataFrame(rows)
    per_asset["bmp_p_holm"] = mt.holm(per_asset["bmp_p"])
    per_asset["bmp_p_bh"] = mt.benjamini_hochberg(per_asset["bmp_p"])
    return per_asset, pd.DataFrame(pooled), fits, cov


def placebo(datas: list[AssetData], calendar: pd.DatetimeIndex, times: pd.Series,
            landfall_times: pd.Series, years: tuple[int, int], n_sets: int, seed: int
            ) -> tuple[dict[str, NDArray], NDArray]:
    """Placebo mean CAR[0, +10] per asset and pooled signed (directional assets).

    Placebo dates are drawn on `calendar` (the market's return dates) by
    realstudy.placebo_day0 and mapped to each asset's calendar (first asset date at or after);
    asset-events whose windows do not fit or that fail the data screen are left out (NaN).
    """
    rng = np.random.default_rng([seed, 1])
    p0 = rs.placebo_day0(times, landfall_times, calendar, years, n_sets, rng)
    out = {}
    for d in datas:
        rows = np.searchsorted(d.dates.to_numpy(), calendar.to_numpy()[p0], side="left")
        ok = np.vectorize(lambda x, d=d: rs.window_fits(int(x), len(d.dates))
                          and rs.quality_ok(d.ret, int(x)))(rows)
        car = np.full(rows.shape, np.nan)
        for c in range(0, n_sets, 100):  # chunks keep memory below a few hundred MB
            sub_ok = ok[c: c + 100]
            block = np.full(sub_ok.shape, np.nan)
            if sub_ok.any():
                block[sub_ok] = rs.car_for_rows(d.ret, d.mkt, rows[c: c + 100][sub_ok], 0, 10)
            car[c: c + 100] = block
        with np.errstate(all="ignore"):
            out[d.asset.name] = np.nanmean(car, axis=1)
    directional = [d.asset for d in datas if d.asset.directional]
    with np.errstate(all="ignore"):
        signed = np.nanmean(np.stack([a.sign * out[a.name] for a in directional]), axis=0) \
            if directional else np.full(n_sets, np.nan)
    return out, signed


def placebo_table(per_asset: pd.DataFrame, pooled: pd.DataFrame, pl: dict[str, NDArray],
                  pl_signed: NDArray) -> pd.DataFrame:
    """Position of each real mean CAR[0, +10] in its placebo distribution."""
    post = per_asset[per_asset.window == "post[0,+10]"].set_index("asset")
    rows = []
    for a in pl:
        real = post.loc[a, "mean_car"] if "mean_car" in post.columns else np.nan
        le, p = rs.placebo_position(real, pl[a])
        rows.append({"series": a, "real_mean_car": real,
                     "placebo_mean": np.nanmean(pl[a]), "placebo_sd": np.nanstd(pl[a], ddof=1),
                     "fraction_placebo_le_real": le, "placebo_p_two_sided": p})
    sel = pooled[(pooled.group == "storm (signed)") & (pooled.window == "post[0,+10]")] \
        if len(pooled) else pooled
    if len(sel):
        real = sel["mean_car"].item()
        le, p = rs.placebo_position(real, pl_signed)
        rows.append({"series": "storm (signed, pooled)", "real_mean_car": real,
                     "placebo_mean": pl_signed.mean(), "placebo_sd": pl_signed.std(ddof=1),
                     "fraction_placebo_le_real": le, "placebo_p_two_sided": p})
    return pd.DataFrame(rows)


def findings(assets: list[Asset], per_asset: pd.DataFrame, pooled: pd.DataFrame,
             placebo_rows: pd.DataFrame) -> pd.DataFrame:
    """The protocol's criteria: Holm-adjusted BMP p < 0.05, placebo p < 0.05 and the
    hypothesised sign (any sign for two-sided assets and controls); pooled: BMP-KP p < 0.05 and
    placebo p < 0.05."""
    post = per_asset[per_asset.window == "post[0,+10]"].set_index("asset")
    pmap = dict(zip(placebo_rows["series"], placebo_rows["placebo_p_two_sided"], strict=True))
    out = []
    for a in assets:
        r = post.loc[a.name]
        if "mean_car" not in post.columns or pd.isna(r["mean_car"]):
            out.append({"series": a.name, "bmp_p_holm": np.nan, "placebo_p": np.nan,
                        "sign_as_hypothesised": None, "meets_criterion": False})
            continue
        sign_ok = (np.sign(r["mean_car"]) == a.sign) if a.directional else True
        hit = bool(r["bmp_p_holm"] < 0.05 and pmap[a.name] < 0.05 and sign_ok)
        out.append({"series": a.name, "bmp_p_holm": r["bmp_p_holm"], "placebo_p": pmap[a.name],
                    "sign_as_hypothesised": bool(sign_ok) if a.directional else None,
                    "meets_criterion": hit})
    sel = pooled[(pooled.group == "storm (signed)") & (pooled.window == "post[0,+10]")] \
        if len(pooled) else pooled
    if len(sel):
        pr = sel.iloc[0]
        pp = pmap["storm (signed, pooled)"]
        out.append({"series": "storm (signed, pooled)", "bmp_p_holm": pr["bmp_kp_p"],
                    "placebo_p": pp, "sign_as_hypothesised": None,
                    "meets_criterion": bool(pr["bmp_kp_p"] < 0.05 and pp < 0.05)})
    return pd.DataFrame(out)


# ---- trading ---------------------------------------------------------------------------------

Entries = Callable[[object, AssetData], tuple[pd.DataFrame, int]]
"""(configuration, asset data) -> (entries with columns event, season, entry_row on the asset's
calendar, in time order; holding period in trading days)."""


def costs_for(asset: Asset, costs: bt.Costs) -> bt.Costs:
    """Futures pay no borrow on short positions (docs/protocol_own_prices.md)."""
    return replace(costs, borrow_bps_per_year=0.0) if asset.kind == "future" else costs


def event_table(trades: pd.DataFrame) -> pd.DataFrame:
    """Mean net return per event (over the assets traded on it)."""
    t = trades.dropna(subset=["net"])
    return t.groupby(["event", "season"], as_index=False)["net"].mean()


def metrics(evt: pd.DataFrame, n_years: int, seed: int) -> dict:
    x = evt["net"].to_numpy()
    if len(x) < 3:
        return {"events": len(x)}
    lo, hi = bs.bootstrap_mean_ci(x, 2.0, 2000, seed)
    sr = sh.sharpe(x)
    g3, g4 = sh.moments(x)
    return {"events": len(x), "mean_net": x.mean(), "boot_lo": lo, "boot_hi": hi,
            "hit_rate": float((x > 0).mean()), "sharpe_per_event": sr,
            "sharpe_annualised": sh.annualise(sr, len(x) / n_years), "skew": g3, "kurt": g4,
            "psr_vs_0": sh.psr(sr, len(x), g3, g4)}


def trades_for(datas: list[AssetData], cfg: object, entries_for: Entries, costs: bt.Costs
               ) -> pd.DataFrame:
    """One market-hedged trade per directional asset and entry (one open position per asset;
    the estimation-window data screen is applied at entry)."""
    parts = []
    for d in datas:
        a = d.asset
        if not a.directional:
            continue
        entries, hold = entries_for(cfg, d)
        n = len(d.mkt)
        rows = []
        for e in entries.itertuples():
            lo = e.entry_row - es.GAP - es.EST_LEN
            if e.entry_row < 0 or e.entry_row + hold >= n or lo < 0:
                continue
            if (d.ret[lo: e.entry_row - es.GAP] == 0).mean() > rs.MAX_ZERO_SHARE:
                continue
            rows.append((e.event, e.season, e.entry_row, d.dates[e.entry_row]))
        t = pd.DataFrame(rows, columns=["event", "season", "entry_row", "entry_date"])
        if t.empty:
            continue
        t = t.sort_values("entry_row", kind="stable").reset_index(drop=True)
        t = t[bt.non_overlapping(t["entry_row"].to_numpy(), np.zeros(len(t), int), hold)]
        t = t.reset_index(drop=True)
        res = bt.trade_returns(d.ret[:, None], d.mkt, t["entry_row"].to_numpy(),
                               np.zeros(len(t), int), np.full(len(t), a.sign), hold,
                               costs_for(a, costs))
        res = res.drop(columns=["asset_col"])
        parts.append(pd.concat([t[["event", "season", "entry_date"]], res], axis=1)
                     .assign(asset=a.name))
    if not parts:
        return pd.DataFrame(columns=["event", "season", "entry_date", "asset", "net"])
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["entry_date", "asset"], kind="stable").reset_index(drop=True)


def evaluate_rule(name: str, configs: list, entries_for: Entries, datas: list[AssetData],
                  costs: bt.Costs, n_years: int, seed: int, primary: object,
                  trials_multiplier: int = 1) -> tuple[pd.DataFrame, dict, dict, dict]:
    """Metrics per configuration, the deflated Sharpe ratio of the best one (N = number of
    configurations x `trials_multiplier`) and the leave-one-season-out evaluation."""
    tables, rows = {}, []
    for cfg in configs:
        evt = event_table(trades_for(datas, cfg, entries_for, costs))
        tables[str(cfg)] = evt
        rows.append({"rule": name, "config": str(cfg), "primary": cfg == primary,
                     **metrics(evt, n_years, seed)})
    table = pd.DataFrame(rows)
    if "sharpe_per_event" not in table.columns or table["sharpe_per_event"].isna().all():
        return table, {"rule": name, "n_trials": len(configs) * trials_multiplier}, \
            {"rule": name, "events": 0}, tables
    srs = table["sharpe_per_event"].to_numpy(dtype=float)
    best = int(np.nanargmax(srs))
    bevt = tables[table.loc[best, "config"]]["net"].to_numpy()
    g3, g4 = sh.moments(bevt)
    var_sr = float(np.nanvar(srs, ddof=1))
    n_trials = len(configs) * trials_multiplier
    dsr = {"rule": name, "best_config": table.loc[best, "config"], "n_trials": n_trials,
           "var_sr_trials": var_sr, "sr_best": srs[best],
           "expected_max_sr": sh.expected_max_sharpe(n_trials, var_sr),
           "events_best": len(bevt),
           "dsr": sh.dsr(srs[best], len(bevt), g3, g4, n_trials, var_sr)}
    loso = bt.leave_one_season_out(tables)
    lrow = {"rule": name, **metrics(loso, n_years, seed),
            "configs_chosen": json.dumps(loso.groupby("season")["config"].first().to_dict())}
    return table, dsr, lrow, tables


def trading_finding(dsr: dict, loso: dict) -> dict:
    """R2 criterion: leave-one-season-out bootstrap interval above 0 and DSR > 0.95."""
    return {"rule": dsr["rule"], "loso_mean_net": loso.get("mean_net"),
            "loso_boot_lo": loso.get("boot_lo"), "dsr_best": dsr.get("dsr"),
            "meets_criterion": bool(loso.get("boot_lo", -1) > 0 and dsr.get("dsr", 0) > 0.95)}


# ---- the two trading rules of the protocols ----------------------------------------------------

R1_GRID = [(a, b) for a in (-5, -3, -1) for b in (3, 5, 10)]
"""R1 calendar rule: (entry day, exit day) relative to the landfall day 0."""
R2_GRID = [(p, h) for p in (0.2, 0.4, 0.6) for h in (5, 10, 15)]
"""R2 forecast-trigger rule: (probability threshold p*, holding period in trading days)."""
R1_PRIMARY, R2_PRIMARY = (-5, 10), (0.4, 10)


def calendar_entries(events: pd.DataFrame) -> Entries:
    """R1 (not tradeable: uses the realised landfall date): enter at the close of day `a`,
    exit at the close of day `b`. `events` has storm_id, season, time, in time order."""
    def entries(cfg, d: AssetData) -> tuple[pd.DataFrame, int]:
        a, b = cfg
        d0 = es.event_rows(d.dates, events["time"])
        return pd.DataFrame({"event": events["storm_id"].to_numpy(),
                             "season": events["season"].to_numpy(), "entry_row": d0 + a}), b - a
    return entries


def trigger_entries(probs: pd.DataFrame) -> Entries:
    """R2 (tradeable): enter at the first close at or after the public time of each storm's
    first forecast with US landfall probability >= p*; hold `h` trading days."""
    from landfall import signals

    def entries(cfg, d: AssetData) -> tuple[pd.DataFrame, int]:
        p_star, hold = cfg
        trig = signals.first_triggers(probs, p_star).sort_values("time")
        rows = rs.entry_rows_from_times(d.dates, trig["available_time"])
        return pd.DataFrame({"event": trig["storm_id"].to_numpy(),
                             "season": trig["season"].to_numpy(), "entry_row": rows}), hold
    return entries
