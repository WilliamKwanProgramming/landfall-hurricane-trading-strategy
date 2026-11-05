"""Point-in-time forecast model: algebra, synthetic tracks and look-ahead guards."""

import numpy as np
import pandas as pd
import pytest

from landfall import forecast as F
from landfall.data.hurdat2 import TRACK_COLUMNS
from landfall.geo import offset_latlon


def synthetic_tracks(n_storms=60, seasons=(1990, 2001), seed=0, steps=40, zonal=False):
    """Storms that move with a constant velocity drawn per storm and a slowly drifting vmax.

    With zonal=True the storms move along a parallel at a constant rate, so that the
    displacement over any interval is exactly proportional to its length.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_storms):
        season = int(rng.integers(*seasons))
        lat0, lon0 = rng.uniform(12, 30), rng.uniform(-80, -40)
        ve, vn = rng.normal(-15, 5), (0.0 if zonal else rng.normal(8, 4))  # km/h
        t0 = pd.Timestamp(f"{season}-08-01") + pd.Timedelta(days=int(rng.integers(0, 60)))
        v0, dv = rng.uniform(30, 120), rng.normal(0, 0.5)
        h = np.arange(steps) * 6.0
        lat, lon = offset_latlon(lat0, lon0, ve * h, vn * h)
        for i in range(steps):
            r = dict.fromkeys(TRACK_COLUMNS, np.nan)
            r.update(storm_id=f"AL{k:02d}{season}", name=f"S{k}", season=season,
                     time=t0 + pd.Timedelta(hours=h[i]), record="", status="HU",
                     lat=lat[i], lon=lon[i], vmax_kt=max(v0 + dv * h[i], 20.0), source="syn")
            rows.append(r)
    df = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    df["time"] = df["time"].astype("datetime64[s]")
    return df


@pytest.fixture(scope="module")
def syn():
    return synthetic_tracks()


def test_ridge_matches_sklearn_and_ols():
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 6)) * [1, 5, 0.1, 2, 3, 10] + [0, 3, 1, 0, -2, 5]
    y = x @ rng.normal(size=6) + rng.normal(size=500)
    f = F.fit_ridge(x, y, lam=3.0)
    z = (x - x.mean(0)) / x.std(0)
    sk = Ridge(alpha=3.0).fit(z, y)
    np.testing.assert_allclose(f.coef, sk.coef_, rtol=1e-10)
    np.testing.assert_allclose(f.predict(x), sk.predict(z), rtol=1e-10)
    f0 = F.fit_ridge(x, y, lam=0.0)
    beta, *_ = np.linalg.lstsq(np.c_[np.ones(500), x], y, rcond=None)
    np.testing.assert_allclose(f0.predict(x), np.c_[np.ones(500), x] @ beta, rtol=1e-9)


def test_feature_construction_by_hand(syn):
    smp = F.build_samples(syn)
    st = syn[syn.storm_id == syn.storm_id.iloc[0]].reset_index(drop=True)
    row = smp[(smp.storm_id == st.storm_id[0]) & (smp.time == st.time[10])].iloc[0]
    from landfall.geo import displacement_km

    e, n = displacement_km(st.lat[8], st.lon[8], st.lat[10], st.lon[10])
    assert row["u12"] == pytest.approx(e) and row["v12"] == pytest.approx(n)
    assert row["dv24"] == pytest.approx(st.vmax_kt[10] - st.vmax_kt[6])
    e48, n48 = displacement_km(st.lat[10], st.lon[10], st.lat[18], st.lon[18])
    assert row["east_48"] == pytest.approx(e48) and row["north_48"] == pytest.approx(n48)
    # The first four fixes have no 24-h history: no samples there.
    assert not ((smp.storm_id == st.storm_id[0]) & (smp.time < st.time[4])).any()
    # Near the end of the storm the long leads have no truth.
    last = smp[(smp.storm_id == st.storm_id[0]) & (smp.time == st.time[39])].iloc[0]
    assert np.isnan(last["east_12"])


def test_persistence_is_exact_for_straight_tracks():
    smp = F.build_samples(synthetic_tracks(zonal=True))
    fc = F.rolling_forecasts(smp, range(1995, 2001))
    err = F.forecast_errors(fc)
    pers = err[err.method == "pers"]
    assert pers["track_km"].max() < 1e-6
    # Persistence of intensity: the error is the observed change.
    d120 = fc[fc["dv_120"].notna()]["dv_120"].abs().to_numpy()
    np.testing.assert_allclose(np.sort(pers[pers.lead_h == 120]["int_abs_kt"]), np.sort(d120))
    ridge = err[err.method == "ridge"].groupby("lead_h")["track_km"].mean()
    clim = err[err.method == "clim"].groupby("lead_h")["track_km"].mean()
    assert (ridge < clim).all()


def test_training_never_includes_test_season(syn):
    smp = F.build_samples(syn)
    m = F.fit_model(smp, 1996)
    assert m.last_season < 1996
    fc = F.rolling_forecasts(smp, range(1996, 1997))
    # Changing every target of seasons >= 1996 must not change a 1996 forecast.
    tampered = smp.copy()
    late = tampered.season >= 1996
    for L in F.LEADS_H:
        for t in F.TARGETS:
            tampered.loc[late, f"{t}_{L}"] += 1000.0
    fc2 = F.rolling_forecasts(tampered, range(1996, 1997))
    cols = [c for c in fc.columns if c.startswith("pred_")]
    np.testing.assert_array_equal(fc[cols].to_numpy(), fc2[cols].to_numpy())


def test_lookahead_guard_on_raw_tracks(syn):
    """Changing or deleting track data after t leaves the forecast made at t unchanged."""
    smp = F.build_samples(syn)
    target = smp[smp.season == 1998].iloc[5]
    t = target["time"]

    def forecast_from(tracks):
        s = F.build_samples(tracks)
        m = F.fit_model(s, 1998)
        row = s[(s.storm_id == target["storm_id"]) & (s.time == t)]
        return np.array([v[0] for v in m.predict_targets(row).values()])

    base = forecast_from(syn)
    later = syn["time"] > t
    moved = syn.copy()
    moved.loc[later, "lat"] += 3.0
    moved.loc[later, "vmax_kt"] += 40.0
    np.testing.assert_array_equal(base, forecast_from(moved))
    np.testing.assert_array_equal(base, forecast_from(syn[~later]))


def test_sample_tracks_zero_residuals_reproduce_forecast(syn):
    smp = F.build_samples(syn)
    m = F.fit_model(smp, 1998)
    row = smp[smp.season == 1998].iloc[0]
    pred = {k: v[0] for k, v in m.predict_targets(row.to_frame().T.astype(
        {c: float for c in F.FEATURES})).items()}
    fc = F.to_track_forecast(row, pred)
    lat, lon, v = F.sample_tracks(fc, np.zeros((1, len(F.LEADS_H), 3)), np.zeros(3, int))
    np.testing.assert_allclose(lat[0], fc.lat, atol=1e-9)
    np.testing.assert_allclose(lon[0], fc.lon, atol=1e-9)
    np.testing.assert_allclose(v[0], fc.vmax_kt, atol=1e-9)


def test_storm_block_bootstrap_reflects_between_storm_spread():
    # Two kinds of storms with very different but internally constant errors: resampling whole
    # storms gives a wide interval; resampling forecasts independently would give a narrow one.
    rng = np.random.default_rng(0)
    rows = []
    for s in range(40):
        level = 100.0 if s % 2 else 300.0
        for _ in range(50):
            rows.append({"storm_id": f"S{s}", "lead_h": 24, "method": "ridge",
                         "track_km": level + rng.normal(0, 1)})
    err = pd.DataFrame(rows)
    ci = F.storm_block_bootstrap_ci(err, "track_km", n_boot=2000, seed=1).iloc[0]
    # sd of the mean over 40 storms with values 100/300: 100 / sqrt(40) = 15.8
    assert 2 * 1.96 * 15.8 * 0.8 < ci["ci_hi"] - ci["ci_lo"] < 2 * 1.96 * 15.8 * 1.2


def test_real_tracks_model_beats_persistence_at_72h(rda_tracks):
    smp = F.build_samples(rda_tracks)
    fc = F.rolling_forecasts(smp, range(2015, 2017))
    err = F.forecast_errors(fc)
    m = err[err.lead_h == 72].groupby("method")["track_km"].mean()
    assert m["ridge"] < m["pers"] and m["ridge"] < m["clim"]
