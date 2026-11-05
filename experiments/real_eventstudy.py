#!/usr/bin/env python3
"""Experiment (d): the pre-specified real-data event study of docs/protocol.md.

Real data: skfolio daily prices of 20 US stocks and the S&P 500 index (1990-2022) and HURDAT2
best tracks. Run once, after the protocol was committed. Writes results/real_eventstudy/.

Usage: python experiments/real_eventstudy.py [--seed 20261105] [--placebo-sets 1000]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from landfall import events as ev
from landfall import forecast as F
from landfall import realstudy as rs
from landfall import signals
from landfall.data import hurdat2, paths
from landfall.data.prices import load_skfolio_prices, simple_returns
from landfall.data.tracks import load_combined_tracks
from landfall.geo import atlantic_land_grid
from landfall.provenance import results_dir, write_manifest
from landfall.study import backtest as bt
from landfall.study import bootstrap as bs
from landfall.study import eventstudy as es
from landfall.study import inference as inf
from landfall.study import multiple as mt
from landfall.study import sharpe as sh

STORM = {"HD": 1, "XOM": -1, "CVX": -1, "RRC": 1}
CONTROLS = ["MSFT", "JNJ", "MRK", "PFE"]
SEASONS = (1990, 2022)
R1_GRID = [(a, b) for a in (-5, -3, -1) for b in (3, 5, 10)]
R2_GRID = [(p, h) for p in (0.2, 0.4, 0.6) for h in (5, 10, 15)]
R1_PRIMARY, R2_PRIMARY = (-5, 10), (0.4, 10)


# ---- event study ----------------------------------------------------------------------------

def study(ret: pd.DataFrame, mkt: np.ndarray, day0: np.ndarray, seed: int) -> tuple[
        pd.DataFrame, pd.DataFrame, dict]:
    """Per-asset tests, pooled tests and the fitted models for the given events."""
    rows, pooled, fits = [], [], {}
    for asset in [*STORM, *CONTROLS]:
        fit, n_screened = rs.fit_asset(ret[asset].to_numpy(), mkt, day0, asset)
        fits[asset] = fit
        for wname, (t1, t2) in es.CAR_WINDOWS.items():
            r = {"asset": asset, "group": "storm" if asset in STORM else "control",
                 "window": wname, "screened_out": n_screened}
            if fit is not None:
                r.update(rs.window_stats(fit.mm.car(t1, t2), fit.mm.scar(t1, t2), fit.mm, t1, t2,
                                         2.0, seed))
            rows.append(r)
    for group, assets, signed in (("storm (signed)", list(STORM), True),
                                  ("control", CONTROLS, False)):
        for wname, (t1, t2) in es.CAR_WINDOWS.items():
            car, scar, ev_id, resid, ar = [], [], [], [], []
            for a in assets:
                f = fits[a]
                if f is None:
                    continue
                s = STORM[a] if signed else 1
                car.append(s * f.mm.car(t1, t2))
                scar.append(s * f.mm.scar(t1, t2))
                ev_id.append(f.events)
                resid.append(s * f.mm.resid_est)
                ar.append(s * f.mm.ar)
            car, scar, ev_id = map(np.concatenate, (car, scar, ev_id))
            resid, ar = np.vstack(resid), np.vstack(ar)
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
    return pd.DataFrame(rows), pd.DataFrame(pooled), fits


def placebo(ret, mkt, dates, events, landfall_times, n_sets, seed):
    rng = np.random.default_rng([seed, 1])
    p0 = rs.placebo_day0(events["time"], landfall_times, dates, (SEASONS[0] + 1, SEASONS[1]),
                         n_sets, rng)
    out = {}
    for a in [*STORM, *CONTROLS]:
        r = ret[a].to_numpy()
        ok = np.vectorize(lambda d, r=r: rs.quality_ok(r, int(d)))(p0)
        car = np.full(p0.shape, np.nan)
        for c in range(0, n_sets, 100):  # chunks keep memory below a few hundred MB
            sub_ok = ok[c: c + 100]
            block = np.full(sub_ok.shape, np.nan)
            block[sub_ok] = rs.car_for_rows(r, mkt, p0[c: c + 100][sub_ok], 0, 10)
            car[c: c + 100] = block
        out[a] = np.nanmean(car, axis=1)
    signed = np.nanmean(np.stack([STORM[a] * out[a] for a in STORM]), axis=0)
    return out, signed


# ---- trading --------------------------------------------------------------------------------

def event_table(trades: pd.DataFrame) -> pd.DataFrame:
    """Mean net return per event (over the stocks traded on it)."""
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


def trades_for(ret, mkt, entries: pd.DataFrame, hold: int, costs: bt.Costs) -> pd.DataFrame:
    """entries: event, season, entry_row (time order). One trade per storm stock and entry."""
    rows = []
    for a, s in STORM.items():
        col = list(ret.columns).index(a)
        r = ret[a].to_numpy()
        for e in entries.itertuples():
            if e.entry_row < 0 or e.entry_row + hold >= len(mkt):
                continue
            est = r[e.entry_row - es.GAP - es.EST_LEN: e.entry_row - es.GAP]
            if e.entry_row - es.GAP - es.EST_LEN < 0 or (est == 0).mean() > rs.MAX_ZERO_SHARE:
                continue
            rows.append((e.event, e.season, e.entry_row, col, s))
    t = pd.DataFrame(rows, columns=["event", "season", "entry_row", "asset_col", "sign"])
    t = t.sort_values(["entry_row", "asset_col"], kind="stable").reset_index(drop=True)
    keep = bt.non_overlapping(t["entry_row"].to_numpy(), t["asset_col"].to_numpy(), hold)
    t = t[keep].reset_index(drop=True)
    res = bt.trade_returns(ret.to_numpy(), mkt, t["entry_row"].to_numpy(),
                           t["asset_col"].to_numpy(), t["sign"].to_numpy(), hold, costs)
    return pd.concat([t[["event", "season"]], res], axis=1)


def evaluate_rule(name, configs, entries_for, ret, mkt, costs, n_years, seed, primary):
    tables, rows = {}, []
    for cfg in configs:
        entries, hold = entries_for(cfg)
        evt = event_table(trades_for(ret, mkt, entries, hold, costs))
        tables[str(cfg)] = evt
        rows.append({"rule": name, "config": str(cfg), "primary": cfg == primary,
                     **metrics(evt, n_years, seed)})
    table = pd.DataFrame(rows)
    srs = table["sharpe_per_event"].to_numpy(dtype=float)
    best = int(np.nanargmax(srs))
    bevt = tables[table.loc[best, "config"]]["net"].to_numpy()
    g3, g4 = sh.moments(bevt)
    var_sr = float(np.nanvar(srs, ddof=1))
    dsr = {"rule": name, "best_config": table.loc[best, "config"], "n_trials": len(configs),
           "var_sr_trials": var_sr, "sr_best": srs[best],
           "expected_max_sr": sh.expected_max_sharpe(len(configs), var_sr),
           "events_best": len(bevt),
           "dsr": sh.dsr(srs[best], len(bevt), g3, g4, len(configs), var_sr)}
    loso = bt.leave_one_season_out(tables)
    lrow = {"rule": name, **metrics(loso, n_years, seed),
            "configs_chosen": json.dumps(loso.groupby("season")["config"].first().to_dict())}
    return table, dsr, lrow, tables


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--placebo-sets", type=int, default=1000)
    args = ap.parse_args()
    out = results_dir("real_eventstudy")

    prices, index = load_skfolio_prices()
    ret = simple_returns(prices)
    mkt = simple_returns(index).to_numpy()
    dates = ret.index

    grid = atlantic_land_grid()
    tracks, _ = load_combined_tracks()
    geo = ev.geometric_landfalls(tracks[tracks.season >= 1985], grid)
    events = ev.storms_with_us_hurricane_landfall(geo)
    events = events[events.season.between(*SEASONS)].sort_values("time").reset_index(drop=True)
    us_all = ev.us_landfalls(geo)  # any intensity, for the placebo exclusion
    day0 = es.event_rows(dates, events["time"])
    fits_ok = np.array([rs.window_fits(int(d), len(dates)) for d in day0])
    keep = np.zeros(len(events), dtype=bool)
    keep[fits_ok] = rs.drop_overlapping(day0[fits_ok])
    events["day0_date"] = [dates[d].date() if d >= 0 else None for d in day0]
    events["windows_fit"] = fits_ok
    events["kept_primary"] = keep
    events.to_csv(out / "events.csv", index=False)
    n_years = SEASONS[1] - SEASONS[0] + 1

    # Primary event study.
    prim = events[keep].reset_index(drop=True)
    per_asset, pooled, _ = study(ret, mkt, day0[keep], args.seed)
    per_asset["bmp_p_holm"] = mt.holm(per_asset["bmp_p"])
    per_asset["bmp_p_bh"] = mt.benjamini_hochberg(per_asset["bmp_p"])
    per_asset.to_csv(out / "car_tests_primary.csv", index=False, float_format="%.6g")
    pooled.to_csv(out / "pooled_primary.csv", index=False, float_format="%.6g")

    # Placebo.
    pl, pl_signed = placebo(ret, mkt, dates, prim, us_all["time"], args.placebo_sets, args.seed)
    prow = []
    post = per_asset[per_asset.window == "post[0,+10]"].set_index("asset")
    for a in [*STORM, *CONTROLS]:
        le, p = rs.placebo_position(post.loc[a, "mean_car"], pl[a])
        prow.append({"series": a, "real_mean_car": post.loc[a, "mean_car"],
                     "placebo_mean": np.nanmean(pl[a]), "placebo_sd": np.nanstd(pl[a], ddof=1),
                     "fraction_placebo_le_real": le, "placebo_p_two_sided": p})
    real_signed = pooled[(pooled.group == "storm (signed)")
                         & (pooled.window == "post[0,+10]")]["mean_car"].item()
    le, p = rs.placebo_position(real_signed, pl_signed)
    prow.append({"series": "storm (signed, pooled)", "real_mean_car": real_signed,
                 "placebo_mean": pl_signed.mean(), "placebo_sd": pl_signed.std(ddof=1),
                 "fraction_placebo_le_real": le, "placebo_p_two_sided": p})
    pd.DataFrame(prow).to_csv(out / "placebo.csv", index=False, float_format="%.6g")

    # Findings by the protocol's criteria.
    find = []
    pmap = {r["series"]: r["placebo_p_two_sided"] for r in prow}
    for a in [*STORM, *CONTROLS]:
        r = post.loc[a]
        sign_ok = (np.sign(r["mean_car"]) == STORM[a]) if a in STORM else True
        hit = bool(r["bmp_p_holm"] < 0.05 and pmap[a] < 0.05 and sign_ok)
        find.append({"series": a, "bmp_p_holm": r["bmp_p_holm"], "placebo_p": pmap[a],
                     "sign_as_hypothesised": bool(sign_ok) if a in STORM else None,
                     "meets_criterion": hit})
    pr = pooled[(pooled.group == "storm (signed)") & (pooled.window == "post[0,+10]")].iloc[0]
    find.append({"series": "storm (signed, pooled)", "bmp_p_holm": pr["bmp_kp_p"],
                 "placebo_p": pmap["storm (signed, pooled)"], "sign_as_hypothesised": None,
                 "meets_criterion": bool(pr["bmp_kp_p"] < 0.05
                                         and pmap["storm (signed, pooled)"] < 0.05)})
    pd.DataFrame(find).to_csv(out / "findings_event_study.csv", index=False,
                              float_format="%.6g")

    # Trading rules.
    costs = bt.Costs()

    def r1_entries(cfg, ev_df=prim, d0=day0[keep]):
        a, b = cfg
        return pd.DataFrame({"event": ev_df["storm_id"], "season": ev_df["season"],
                             "entry_row": d0 + a}), b - a

    samples = F.build_samples(tracks)
    probs = signals.landfall_probabilities(samples, range(SEASONS[0], SEASONS[1] + 1),
                                           atlantic_land_grid(coarsen=6), 200, args.seed)
    probs.to_csv(out / "landfall_probabilities.csv.gz", index=False, float_format="%.4f",
                 compression={"method": "gzip", "mtime": 0})

    def r2_entries(cfg):
        p_star, hold = cfg
        trig = signals.first_triggers(probs, p_star).sort_values("time")
        rows = rs.entry_rows_from_times(dates, trig["available_time"])
        return pd.DataFrame({"event": trig["storm_id"], "season": trig["season"],
                             "entry_row": rows}), hold

    tabs, dsrs, losos = [], [], []
    for name, grid_, fn, prim_cfg in (("R1 calendar (not tradeable)", R1_GRID, r1_entries,
                                       R1_PRIMARY),
                                      ("R2 forecast trigger", R2_GRID, r2_entries, R2_PRIMARY)):
        t, d, lrow, _ = evaluate_rule(name, grid_, fn, ret, mkt, costs, n_years, args.seed,
                                      prim_cfg)
        tabs.append(t)
        dsrs.append(d)
        losos.append(lrow)
    pd.concat(tabs).to_csv(out / "trading_configs.csv", index=False, float_format="%.6g")
    pd.DataFrame(dsrs).to_csv(out / "trading_dsr.csv", index=False, float_format="%.6g")
    pd.DataFrame(losos).to_csv(out / "trading_loso.csv", index=False, float_format="%.6g")
    r2_dsr = dsrs[1]
    r2_loso = losos[1]
    pd.DataFrame([{"rule": "R2 forecast trigger",
                   "loso_mean_net": r2_loso.get("mean_net"), "loso_boot_lo": r2_loso.get(
                       "boot_lo"), "dsr_best": r2_dsr["dsr"],
                   "meets_criterion": bool(r2_loso.get("boot_lo", -1) > 0
                                           and r2_dsr["dsr"] > 0.95)}]).to_csv(
        out / "findings_trading.csv", index=False)

    # Sensitivities.
    sens = []
    all_d0 = day0[fits_ok]
    pa, po, _ = study(ret, mkt, all_d0, args.seed)
    pa["sensitivity"], po["sensitivity"] = "all events kept", "all events kept"
    sens.append((pa, po))
    rda, _ = hurdat2.load_rda_mirror(paths.raw_file(paths.RDA_MIRROR))
    flag_ev = ev.storms_with_us_hurricane_landfall(ev.flagged_landfalls(rda))
    flag_ev = flag_ev[flag_ev.season.between(1990, 2019)].sort_values("time")
    fd0 = es.event_rows(dates, flag_ev["time"])
    fok = np.array([rs.window_fits(int(d), len(dates)) for d in fd0])
    fkeep = rs.drop_overlapping(fd0[fok])
    pa, po, _ = study(ret, mkt, fd0[fok][fkeep], args.seed)
    pa["sensitivity"], po["sensitivity"] = "flag events 1990-2019", "flag events 1990-2019"
    sens.append((pa, po))
    pd.concat([s[0] for s in sens]).to_csv(out / "sensitivity_car_tests.csv", index=False,
                                           float_format="%.6g")
    pd.concat([s[1] for s in sens]).to_csv(out / "sensitivity_pooled.csv", index=False,
                                           float_format="%.6g")
    double = bt.Costs(20.0, 4.0, 100.0)
    crow = []
    for name, cfg, fn in (("R1 calendar (not tradeable)", R1_PRIMARY, r1_entries),
                          ("R2 forecast trigger", R2_PRIMARY, r2_entries)):
        entries, hold = fn(cfg)
        evt = event_table(trades_for(ret, mkt, entries, hold, double))
        crow.append({"rule": name, "config": str(cfg), "costs": "doubled",
                     **metrics(evt, n_years, args.seed)})
    pd.DataFrame(crow).to_csv(out / "sensitivity_costs.csv", index=False, float_format="%.6g")

    write_manifest(out, args.seed, {
        "data": "REAL: skfolio prices 1990-2022, HURDAT2 tracks (GitHub mirrors)",
        "protocol": "docs/protocol.md", "placebo_sets": args.placebo_sets,
        "events_total": len(events), "events_primary": int(keep.sum()),
        "events_dropped_overlap": int(fits_ok.sum() - keep.sum()),
        "events_outside_price_data": int((~fits_ok).sum())})
    print(per_asset[per_asset.window == "post[0,+10]"].to_string())
    print(pooled.to_string())
    print(pd.DataFrame(prow).to_string())
    print(pd.concat(tabs).to_string())
    print(pd.DataFrame(dsrs).to_string())
    print(pd.DataFrame(losos).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
