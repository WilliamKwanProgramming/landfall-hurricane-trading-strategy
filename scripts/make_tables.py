#!/usr/bin/env python3
"""Generate the Markdown tables of the README from the files in results/.

Writes results/tables.md and replaces, in README.md, the text between the markers
<!-- table:KEY --> and <!-- /table:KEY --> with the generated table KEY, so that README tables are
never typed by hand. Every table title says whether its data are real or synthetic.

Usage: python scripts/make_tables.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[1]
R = REPO / "results"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def ci(m: float, lo: float, hi: float, fmt: str) -> str:
    return f"{m:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def forecast_table(name: str, fmt: str) -> pd.DataFrame:
    d = pd.read_csv(R / "forecast_skill" / name)
    rows = []
    for lead, g in d.groupby("lead_h"):
        g = g.set_index("method")
        rows.append({
            "lead (h)": lead, "forecasts": int(g.loc["ridge", "n"]),
            "storms": int(g.loc["ridge", "storms"]),
            "model": ci(*g.loc["ridge", ["mean", "ci_lo", "ci_hi"]], fmt),
            "persistence": ci(*g.loc["pers", ["mean", "ci_lo", "ci_hi"]], fmt),
            "climatology": ci(*g.loc["clim", ["mean", "ci_lo", "ci_hi"]], fmt),
            "model - persistence": ci(*g.loc["ridge-pers", ["mean", "ci_lo", "ci_hi"]], fmt)})
    return pd.DataFrame(rows)


def main() -> int:
    parts: dict[str, str] = {}

    def add(key: str, title: str, df: pd.DataFrame) -> None:
        parts[key] = f"**{title}**\n\n{md(df)}\n"

    # Data checks (real).
    c = json.loads((R / "data_checks" / "mirror_comparison.json").read_text())
    add("mirrors",
        "Mirror comparison, HURDAT2 1975-2019 (real data)", pd.DataFrame([{
            "storms rda": c["storms_rda"], "storms dplyr": c["storms_dplyr"],
            "only in rda: no pressure / never TD-HU": (
                f"{c['storms_only_rda_all_pressure_missing']} / "
                f"{c['storms_only_rda_no_td_ts_hu']}"),
            "dplyr fixes matched": f"{c['fixes_matched']} of {c['fixes_dplyr']}",
            "differing lat / lon / wind / pressure / status": (
                f"{c['diff_lat']} / {c['diff_lon']} / {c['diff_vmax_kt']} / "
                f"{c['diff_mslp_mb']} / {c['diff_status']}")}]))

    # Landfall definitions (real).
    d = pd.read_csv(R / "landfall_counts" / "definitions.csv")
    d = d.assign(share_flag_matched=d["share_flag_matched"].map(lambda x: f"{x:.3f}"),
                 median_hours_apart=d["median_hours_apart"].map(lambda x: f"{x:.2f}"),
                 median_km_apart=d["median_km_apart"].map(lambda x: f"{x:.1f}"))
    add("definitions",
        "Landfall definitions compared, HURDAT2 rda mirror (real data)", d)
    s = pd.read_csv(R / "landfall_counts" / "storm_level_agreement_1851_2019.csv")
    s.columns = ["quantity", "storms"]
    add("storm_level",
        "Storms with a US hurricane landfall, 1851-2019, by definition (real data)", s)
    disp = pd.read_csv(R / "landfall_counts" / "dispersion.csv").round(3)
    add("dispersion",
        "US hurricane-landfall storms per season: Poisson dispersion test (real data)", disp)

    # Forecast skill (real).
    add("track_error",
        "Mean track error (km) by lead, out of sample 1990-2024, 95% storm-block bootstrap "
        "intervals (real data)", forecast_table("track_error_km.csv", ".0f"))
    add("intensity_error",
        "Mean absolute intensity error (kt) by lead, out of sample 1990-2024, 95% storm-block "
        "bootstrap intervals (real data)", forecast_table("intensity_error_kt.csv", ".1f"))

    # Synthetic event study.
    tests = ["naive_t", "patell", "bmp", "bmp_kp", "rank"]
    rows = []
    for sc in ("gaussian", "garch_t", "garch_t_event_variance", "garch_t_cross_correlation"):
        t = pd.read_csv(R / "synthetic_eventstudy" / f"{sc}.csv")
        z = t[t["delta"] == 0].iloc[0]
        row = {"scenario": sc}
        for k in tests:
            row[k] = f"{100 * z[f'reject_{k}']:.1f} ({100 * z[f'reject_{k}_mcse']:.1f})"
        row["t-interval coverage"] = f"{100 * z['t_cover']:.1f} ({100 * z['t_cover_mcse']:.1f})"
        row["bootstrap coverage"] = (f"{100 * z['boot_cover']:.1f} "
                                     f"({100 * z['boot_cover_mcse']:.1f})")
        rows.append(row)
    add("synthetic_size",
        "Size at delta = 0: rejection rate at 5% and interval coverage, % (Monte Carlo SE), "
        "2000 replications, 30 events x 4 stocks (synthetic data)", pd.DataFrame(rows))
    rows = []
    for sc, label in (("garch_t", "30 x 4"), ("garch_t_35x1", "35 x 1")):
        t = pd.read_csv(R / "synthetic_eventstudy" / f"{sc}.csv")
        for _, z in t.iterrows():
            rows.append({"design": label, "delta": pct(z["delta"]),
                         "bias of mean CAR": f"{z['bias']:+.5f} ({z['bias_mcse']:.5f})",
                         **{k: f"{100 * z[f'reject_{k}']:.1f}" for k in tests}})
    add("synthetic_power",
        "Power under GARCH-t: rejection rate at 5%, % (bias with Monte Carlo SE), 2000 "
        "replications (synthetic data)", pd.DataFrame(rows))

    # Real event study.
    a = pd.read_csv(R / "real_eventstudy" / "car_tests_primary.csv")
    pl = pd.read_csv(R / "real_eventstudy" / "placebo.csv").set_index("series")
    post = a[a["window"] == "post[0,+10]"]
    rows = []
    for _, z in post.iterrows():
        rows.append({"stock": z["asset"], "group": z["group"], "events": int(z["n"]),
                     "mean CAR[0,+10]": ci(100 * z["mean_car"], 100 * z["boot_lo"],
                                           100 * z["boot_hi"], ".2f"),
                     "BMP t": f"{z['bmp_t']:.2f}", "BMP p": f"{z['bmp_p']:.3f}",
                     "Holm p": f"{z['bmp_p_holm']:.3f}", "rank p": f"{z['rank_p']:.3f}",
                     "placebo p": f"{pl.loc[z['asset'], 'placebo_p_two_sided']:.3f}"})
    add("real_car",
        "CAR[0,+10] around US hurricane landfalls 1990-2022, %, 95% stationary-bootstrap "
        "interval (real data)", pd.DataFrame(rows))
    p = pd.read_csv(R / "real_eventstudy" / "pooled_primary.csv")
    p = p.assign(mean_car=lambda x: [ci(100 * m, 100 * lo, 100 * hi, ".2f") for m, lo, hi in
                                     zip(x["mean_car"], x["boot_lo"], x["boot_hi"],
                                         strict=True)])
    p = p[["group", "window", "n", "mean_car", "bmp_t", "bmp_p", "rbar", "bmp_kp_p",
           "rank_p"]].round(3).rename(columns={"mean_car": "mean CAR, % [95% CI]"})
    add("real_pooled",
        "Pooled tests, storm-relevant stocks signed by hypothesis (real data)", p)

    tc = pd.read_csv(R / "real_eventstudy" / "trading_configs.csv")
    tc = tc[tc["primary"]]
    lo = pd.read_csv(R / "real_eventstudy" / "trading_loso.csv")
    ds = pd.read_csv(R / "real_eventstudy" / "trading_dsr.csv").set_index("rule")
    rows = []
    for _, z in tc.iterrows():
        rows.append({"rule": z["rule"], "evaluation": f"primary {z['config']}",
                     "events": int(z["events"]),
                     "mean net per event": ci(100 * z["mean_net"], 100 * z["boot_lo"],
                                              100 * z["boot_hi"], ".2f"),
                     "Sharpe per event": f"{z['sharpe_per_event']:.3f}",
                     "PSR(0)": f"{z['psr_vs_0']:.3f}", "DSR of best of 9": ""})
    for _, z in lo.iterrows():
        rows.append({"rule": z["rule"], "evaluation": "leave-one-season-out",
                     "events": int(z["events"]),
                     "mean net per event": ci(100 * z["mean_net"], 100 * z["boot_lo"],
                                              100 * z["boot_hi"], ".2f"),
                     "Sharpe per event": f"{z['sharpe_per_event']:.3f}",
                     "PSR(0)": f"{z['psr_vs_0']:.3f}",
                     "DSR of best of 9": f"{ds.loc[z['rule'], 'dsr']:.3f} "
                                         f"(best {ds.loc[z['rule'], 'best_config']})"})
    add("trading",
        "Trading rules, net of costs, %, 95% stationary-bootstrap interval (real data)",
        pd.DataFrame(rows))

    bench = R / "benchmark" / "footprint_speed.csv"
    if bench.exists():
        new = pd.read_csv(bench, dtype=str, keep_default_na=False)
        new.insert(0, "measurement", "wave 2")
        old = pd.read_csv(R / "benchmark" / "wave1" / "footprint_speed.csv", dtype=str)
        old["engine"] = old["engine"].replace({"cpp": "cpp (wave-1 kernel)",
                                               "ratio numpy/cpp": "ratio numpy / cpp (wave-1)"})
        old.insert(0, "measurement", "wave 1")
        old["ns_per_evaluation"] = ""
        df = pd.concat([old, new], ignore_index=True)
        df.columns = ["measurement", "engine", "reps", "median (s)", "min (s)", "max (s)",
                      "ns per wind evaluation"]
        add("benchmark",
            "Footprint Monte Carlo speed, 500 storms x 241 steps x 50 points, single thread, "
            "shared machine, indicative (synthetic inputs)", df)

    own = R / "own_prices_synthetic"
    if (own / "car_tests.csv").exists():
        car = pd.read_csv(own / "car_tests.csv")
        car = car[car.window == "post[0,+10]"].set_index("asset")
        fnd = pd.read_csv(own / "findings_event_study.csv").set_index("series")
        cov = pd.read_csv(own / "coverage.csv").set_index("asset")
        rows = []
        for a in car.index:
            r = car.loc[a]
            rows.append({"asset": a, "group": r["group"], "events used": int(cov.loc[a, "used"]),
                         "mean CAR[0,+10], %": ci(100 * r["mean_car"], 100 * r["boot_lo"],
                                                   100 * r["boot_hi"], ".2f"),
                         "Holm p": f"{r['bmp_p_holm']:.3f}",
                         "placebo p": f"{fnd.loc[a, 'placebo_p']:.3f}",
                         "finding": "yes" if fnd.loc[a, "meets_criterion"] else "no"})
        man = dict(ln.split(": ", 1) for ln in (own / "MANIFEST.txt").read_text().splitlines()
                   if ": " in ln)
        add("own_prices_synthetic",
            f"scripts/run_own_prices.py on SYNTHETIC prices without any injected effect: "
            f"simulated series that carry the template asset names, not market prices "
            f"({len(rows)} series, {man['events_primary']} events, "
            f"{man['placebo_sets']} placebo sets, {man['elapsed_seconds']} s; synthetic data)",
            pd.DataFrame(rows))

    ps = R / "placebo_size" / "summary.csv"
    if ps.exists():
        d = pd.read_csv(ps)
        man = dict(ln.split(": ", 1) for ln in (R / "placebo_size" / "MANIFEST.txt").read_text()
                   .splitlines() if ": " in ln)
        d = d.assign(**{"rejection rate, % (MC SE)": [
            f"{100 * r:.1f} ({100 * e:.1f})" for r, e in zip(d.rejection_rate, d.mc_se,
                                                              strict=True)]})
        add("placebo_size",
            f"Size of the placebo test and of the BMP test on null synthetic prices "
            f"({man['reps']} replications x {man['assets_per_rep']} assets, 39 events, "
            f"{man['placebo_sets']} placebo sets; synthetic data)",
            d[["test", "level", "tests", "rejection rate, % (MC SE)"]])

    mut = R / "mutation_check.txt"
    if mut.exists():
        lines = mut.read_text().splitlines()
        res = []
        for ln in lines:
            for tag in (": caught", ": SURVIVED"):
                if tag in ln:
                    res.append((ln.split(tag)[0], tag[2:]))
        last = [ln for ln in lines if "mutants caught" in ln][-1]
        survivors = [m for m, r in res if r == "SURVIVED"]
        parts["mutation"] = (
            f"**Mutation check of the test suite (tests/mutation_check.py): {last}**\n\n"
            + "; ".join(m for m, _ in res) + ".\n"
            + (f"\nSurvivors: {'; '.join(survivors)}.\n" if survivors else ""))

    text = "<!-- Generated by scripts/make_tables.py from results/; do not edit. -->\n\n"
    text += "\n".join(parts.values())
    (R / "tables.md").write_text(text)
    readme = REPO / "README.md"
    body = readme.read_text()
    for key, table in parts.items():
        pat = re.compile(rf"(<!-- table:{key} -->\n).*?(<!-- /table:{key} -->)", re.S)
        body = pat.sub(lambda m, t=table: m.group(1) + t + m.group(2), body)
    readme.write_text(body)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
