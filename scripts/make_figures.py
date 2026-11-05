#!/usr/bin/env python3
"""Draw the README figures from the files in results/ (nothing is recomputed here).

Writes docs/img/*.png at 2x resolution (1,600 px wide). Colours: the first three slots of a
colour-blind-checked categorical palette (blue, orange, aqua) and grey for context; every line
is labelled directly. Usage: python scripts/make_figures.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
R = REPO / "results"
OUT = REPO / "docs" / "img"
BLUE, ORANGE, AQUA, GREY, INK, MUTED = ("#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b",
                                        "#52514e")
DPI = 200

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation":
    "left", "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 2,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / name
    fig.savefig(f, dpi=DPI, bbox_inches="tight", pil_kwargs={"optimize": True})
    plt.close(fig)
    print(f"{f.relative_to(REPO)}: {f.stat().st_size / 1024:.0f} KB")


def label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=INK,
                va="center", fontsize=9)
    ax.plot([x], [y], "o", color=color, ms=4)


def fig_forecast_skill():
    d = pd.read_csv(R / "forecast_skill" / "track_error_km.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for m, name, c in (("ridge", "statistical model", BLUE), ("pers", "persistence", ORANGE),
                       ("clim", "climatology", AQUA)):
        g = d[d.method == m].sort_values("lead_h")
        ax.fill_between(g.lead_h, g.ci_lo, g.ci_hi, color=c, alpha=0.2, lw=0)
        ax.plot(g.lead_h, g["mean"], color=c)
        label_end(ax, g.lead_h.iloc[-1], g["mean"].iloc[-1], name, c)
    ax.set_xticks([12, 24, 48, 72, 96, 120])
    ax.set_xlim(0, 150)
    ax.set_xlabel("forecast lead (h)")
    ax.set_ylabel("mean track error (km)")
    ax.set_title("Track error by lead, out of sample 1990-2024 (real tracks)")
    ax.text(0.01, 0.97, "bands: 95% bootstrap intervals, resampling whole storms",
            transform=ax.transAxes, color=MUTED, fontsize=8.5, va="top")
    save(fig, "forecast_skill.png")


def fig_landfall_counts():
    d = pd.read_csv(R / "landfall_counts" / "us_hurricane_storms_per_season.csv")
    d = d[d.season.between(1900, 2019)]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axvspan(1965.5, 1980.5, color="#f1efe9", lw=0)
    ax.text(1973, 40, "1966-1980: almost no\nlandfall flags\nin the mirror", ha="center",
            va="bottom", fontsize=8.5, color=MUTED)
    geo, flag = d.geometric_1900_2019.cumsum(), d.flag_1900_2019.cumsum()
    ax.plot(d.season, geo, color=BLUE)
    ax.plot(d.season, flag, color=ORANGE, ls=(0, (4, 2)))
    label_end(ax, 2019, geo.iloc[-1], f"geometric crossing ({int(geo.iloc[-1])})", BLUE, dy=-6)
    label_end(ax, 2019, flag.iloc[-1], f'HURDAT2 "L" flag ({int(flag.iloc[-1])})', ORANGE,
              dy=6)
    ax.set_xlim(1898, 2052)
    ax.set_ylim(0, None)
    ax.set_ylabel("storms since 1900 (cumulative)")
    ax.set_title("Storms with a US hurricane landfall, two definitions, 1900-2019 (real tracks)")
    save(fig, "landfall_counts.png")


def fig_car_paths():
    d = pd.read_csv(R / "figure_data" / "car_paths.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(-0.5, color=GREY, lw=0.8, ls=":")
    ax.text(-0.35, 1.45, "landfall (day 0 close)", fontsize=8.5, color=MUTED, va="top")
    for g, name, c in (("storm (signed)", "storm-relevant, signed by hypothesis", BLUE),
                       ("control", "controls", ORANGE)):
        s = d[d.group == g]
        ax.fill_between(s.day, 100 * s.boot_lo, 100 * s.boot_hi, color=c, alpha=0.13, lw=0)
        ax.plot(s.day, 100 * s.mean_car, color=c, marker="o", ms=3.5)
        label_end(ax, 10, 100 * s.mean_car.iloc[-1], f"{name}\n(n = {s.n.iloc[0]})", c,
                  dy=9 if g == "storm (signed)" else -9)
    ax.set_xlim(-5.5, 16.5)
    ax.set_xticks(range(-5, 11))
    ax.set_xlabel("trading day relative to landfall")
    ax.set_ylabel("mean CAR from day -5 (%)")
    ax.set_title("Cumulative abnormal return around 39 US hurricane landfalls (real prices)")
    ax.text(0.01, 0.03, "bands: 95% stationary-bootstrap intervals (2,000 resamples)",
            transform=ax.transAxes, color=MUTED, fontsize=8.5)
    save(fig, "car_paths.png")


def fig_placebo():
    p = pd.read_csv(R / "figure_data" / "placebo_pooled_signed.csv")["pooled_signed_mean_car"]
    pl = pd.read_csv(R / "real_eventstudy" / "placebo.csv").set_index("series")
    real = pl.loc["storm (signed, pooled)", "real_mean_car"]
    pval = pl.loc["storm (signed, pooled)", "placebo_p_two_sided"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(100 * p, bins=40, color=GREY, alpha=0.75, edgecolor="white", linewidth=0.6)
    ax.axvline(100 * real, color=BLUE, lw=2)
    ax.annotate(f"real value {100 * real:.2f}%\nplacebo p = {pval:.2f}",
                (100 * real, ax.get_ylim()[1] * 0.9), xytext=(8, 0),
                textcoords="offset points", color=INK, fontsize=9, va="top",
                bbox={"boxstyle": "square,pad=0.3", "fc": "white", "ec": "none"})
    ax.set_xlabel("pooled signed mean CAR[0, +10] (%)")
    ax.set_ylabel("placebo sets")
    ax.set_title(f"Placebo distribution: {len(p):,} sets of non-landfall dates (real prices)")
    ax.grid(axis="x", visible=False)
    save(fig, "placebo.png")


def fig_footprint():
    from landfall.exposure import load_exposure_map
    from landfall.geo import atlantic_land_grid

    f = pd.read_csv(R / "figure_data" / "footprint_katrina_field.csv")
    t = pd.read_csv(R / "figure_data" / "footprint_katrina_tracks.csv")
    lats, lons = np.sort(f.lat.unique()), np.sort(f.lon.unique())
    shape = (len(lats), len(lons))
    g = atlantic_land_grid(coarsen=6)
    la = g.lat0 + g.dlat * (np.arange(g.land.shape[0]) + 0.5)
    lo = g.lon0 + g.dlon * (np.arange(g.land.shape[1]) + 0.5)
    li = (la >= lats.min() - 1) & (la <= 37)
    lj = (lo >= -99) & (lo <= -80)
    emap = load_exposure_map("gulf_energy_weighted")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.9), sharey=True)
    panels = (("central_track_max_wind_kt", "maximum wind, central forecast track (kt)",
               [34, 50, 64, 83, 96, 113, 137], "Blues"),
              ("p_ge_64kt", "probability of at least 64 kt (200 sampled tracks)",
               [0.05, 0.2, 0.4, 0.6, 0.8, 1.0001], "Oranges"))
    for ax, (col, title, levels, cmap) in zip(axes, panels, strict=True):
        ax.contourf(lo[lj], la[li], g.land[np.ix_(li, lj)], levels=[0.5, 1.5],
                    colors=["#ecebe6"])
        ax.contour(lo[lj], la[li], g.land[np.ix_(li, lj)], levels=[0.5], colors=[GREY],
                   linewidths=0.5)
        z = f[col].to_numpy().reshape(shape)
        cs = ax.contourf(lons, lats, z, levels=levels, cmap=cmap, alpha=0.85, extend="max"
                         if col.startswith("central") else "neither")
        cb = fig.colorbar(cs, ax=ax, orientation="horizontal", pad=0.12, fraction=0.05)
        cb.ax.tick_params(labelsize=8)
        cb.set_label(title, fontsize=8.5)
        for kind, c, ls in (("best track", INK, "--"), ("forecast (central)", BLUE, "-")):
            k = t[(t.kind == kind) & t.lead_h.between(0, 120)]
            ax.plot(k.lon, k.lat, color=c, ls=ls, lw=1.5)
        ax.scatter(emap.lon, emap.lat, s=14, marker="s", facecolor="white", edgecolor=INK,
                   linewidth=0.8, zorder=5)
        ax.set_xlim(-98, -81)
        ax.set_ylim(lats.min(), 36)
        ax.set_aspect(1 / np.cos(np.radians(30)))
        ax.grid(False)
        ax.tick_params(labelsize=8)
    axes[0].text(-97.6, 35.4, "dashed: best track\nsolid: forecast", fontsize=8, va="top",
                 color=INK)
    axes[1].text(-97.6, 35.4, "squares: illustrative\nexposure points (invented)",
                 fontsize=8, va="top", color=INK)
    fig.suptitle("Katrina 2005, statistical forecast issued 27 Aug 12 UTC: wind footprint "
                 "(model output)", x=0.02, y=0.98, ha="left", fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.9)
    save(fig, "footprint_katrina.png")


def fig_size_power():
    tests = [("naive_t", "naive t"), ("patell", "Patell"), ("bmp", "BMP"),
             ("bmp_kp", "BMP-KP"), ("rank", "rank")]
    scen = [("gaussian", "Gaussian"), ("garch_t", "GARCH-t"),
            ("garch_t_event_variance", "event-induced\nvariance"),
            ("garch_t_cross_correlation", "cross-\ncorrelated")]
    fig = plt.figure(figsize=(8, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.6, wspace=0.12)
    for k, (s, title) in enumerate(scen):
        ax = fig.add_subplot(gs[0, k])
        d = pd.read_csv(R / "synthetic_eventstudy" / f"{s}.csv")
        d = d[d.delta == 0].iloc[0]
        y = np.arange(len(tests))
        v = [100 * d[f"reject_{t}"] for t, _ in tests]
        e = [196 * d[f"reject_{t}_mcse"] for t, _ in tests]
        colors = [ORANGE if x > 7.5 else BLUE for x in v]
        ax.barh(y, v, xerr=e, color=colors, height=0.6, error_kw={"lw": 0.8,
                                                                  "ecolor": MUTED})
        ax.axvline(5, color=INK, lw=0.8, ls=":")
        ax.set_xlim(0, 40)
        ax.set_yticks(y, [n for _, n in tests] if k == 0 else [])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9.5, loc="center")
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", visible=False)
        if k == 0:
            ax.text(0, -1.9, "Size: rejection rate of a true null at 5% (%), "
                    "2,000 replications, 30 events x 4 stocks; orange: above 7.5%",
                    fontsize=9.5, fontweight="bold", transform=ax.transData)
    ax = fig.add_subplot(gs[1, :])
    for f, name, c in (("garch_t", "30 events x 4 stocks", BLUE),
                       ("garch_t_35x1", "35 events x 1 stock (a per-stock real-data test)",
                        ORANGE)):
        d = pd.read_csv(R / "synthetic_eventstudy" / f"{f}.csv")
        x, y = 100 * d.delta, 100 * d.reject_bmp
        ax.errorbar(x, y, yerr=196 * d.reject_bmp_mcse, color=c, marker="o", ms=4, capsize=0)
        label_end(ax, x.iloc[-1], y.iloc[-1], name, c)
    ax.axhline(5, color=INK, lw=0.8, ls=":")
    ax.set_xlim(-0.1, 3.6)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 0.5, 1, 2])
    ax.set_xlabel("true mean CAR over 11 days (%)")
    ax.set_ylabel("BMP rejections (%)")
    ax.set_title("Power of the BMP test under GARCH-t returns (synthetic data)")
    save(fig, "size_power.png")


def fig_trading():
    d = pd.read_csv(R / "real_eventstudy" / "trading_configs.csv")
    lo = pd.read_csv(R / "real_eventstudy" / "trading_loso.csv").set_index("rule")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.0), sharex=True)
    for ax, (rule, title, cfgname) in zip(axes, (
            ("R2 forecast trigger", "R2 forecast trigger (tradeable)", "(p*, holding days)"),
            ("R1 calendar (not tradeable)", "R1 calendar (uses the landfall date)",
             "(entry day, exit day)")), strict=True):
        g = d[d.rule == rule].reset_index(drop=True)
        y = np.arange(len(g))
        for i, r in g.iterrows():
            c = BLUE if r.primary else GREY
            ax.plot([100 * r.boot_lo, 100 * r.boot_hi], [i, i], color=c, lw=1.5)
            ax.plot(100 * r.mean_net, i, "o", color=c, ms=5)
        L = lo.loc[rule]
        j = len(g) + 0.8
        ax.plot([100 * L.boot_lo, 100 * L.boot_hi], [j, j], color=ORANGE, lw=1.5)
        ax.plot(100 * L.mean_net, j, "D", color=ORANGE, ms=5)
        labels = [f"{c}{' primary' if p else ''}" for c, p in zip(g.config, g.primary,
                                                                   strict=True)]
        ax.set_yticks([*y, j], [*labels, "leave one\nseason out"])
        ax.axvline(0, color=INK, lw=0.8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(cfgname, fontsize=8.5, color=MUTED)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("mean net return per event (%)")
    fig.suptitle("Trading rules net of costs, 95% bootstrap intervals (real prices)", x=0.02,
                 ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "trading_rules.png")


def main() -> int:
    fig_forecast_skill()
    fig_landfall_counts()
    fig_car_paths()
    fig_placebo()
    fig_footprint()
    fig_size_power()
    fig_trading()
    big = [p.name for p in OUT.glob("*.png") if p.stat().st_size > 300 * 1024]
    if big:
        print(f"over 300 KB: {big}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
