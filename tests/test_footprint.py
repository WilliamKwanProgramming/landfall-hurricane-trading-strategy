"""Wind footprint: hand calculations, limiting cases, and the C++ kernel against NumPy."""

import numpy as np
import pytest
from conftest import cpp_module

from landfall import footprint as fp
from landfall.geo import MS_PER_KT, LandGrid, offset_latlon


def test_profile_equals_vmax_at_rmw_and_peaks_there():
    assert fp.symmetric_profile(35.0, 110.0, 35.0, 1.6) == 110.0
    r = np.linspace(1, 300, 30000)
    v = fp.symmetric_profile(r, 110.0, 35.0, 1.6)
    assert abs(r[np.argmax(v)] - 35.0) < 0.02
    assert fp.symmetric_profile(0.0, 110.0, 35.0, 1.6) == 0.0
    assert fp.symmetric_profile(1e5, 110.0, 35.0, 1.6) < 1.0


def test_profile_by_hand():
    # B = 1, r = 2 Rm: x = 1/2, V = Vm sqrt(0.5 e^0.5).
    assert fp.symmetric_profile(60.0, 100.0, 30.0, 1.0) == pytest.approx(
        100.0 * np.sqrt(0.5 * np.exp(0.5)), rel=1e-14)
    # B = 2, r = Rm / 2: x = 4, V = Vm sqrt(4 e^-3).
    assert fp.symmetric_profile(15.0, 100.0, 30.0, 2.0) == pytest.approx(
        100.0 * np.sqrt(4 * np.exp(-3.0)), rel=1e-14)


def test_holland_b_by_hand():
    vg = 100 * MS_PER_KT / 0.9  # 57.16 m/s
    expected = 1.15 * np.e * vg ** 2 / ((1013 - 950) * 100.0)
    assert expected == pytest.approx(1.6213, abs=1e-4)
    assert fp.holland_b(100.0, 950.0) == pytest.approx(expected, rel=1e-14)
    assert fp.holland_b(150.0, 1010.0) == 2.5  # clipped
    assert fp.holland_b(30.0, 900.0) == 1.0  # clipped
    assert fp.holland_b(80.0, np.nan) == 1.31 and fp.holland_b(80.0, 1015.0) == 1.31


def test_rmax_willoughby_by_hand():
    vg = 90 * MS_PER_KT / 0.9
    assert fp.rmax_willoughby_km(90.0, 25.0) == pytest.approx(
        46.4 * np.exp(-0.0155 * vg + 0.0169 * 25), rel=1e-14)
    assert fp.rmax_willoughby_km(90.0, 25.0) == pytest.approx(31.9, abs=0.1)
    assert fp.rmax_willoughby_km(90.0, 35.0) > fp.rmax_willoughby_km(90.0, 25.0)
    assert fp.rmax_willoughby_km(130.0, 25.0) < fp.rmax_willoughby_km(90.0, 25.0)


def test_rmax_from_r34_closed_form_matches_root_finding():
    from scipy.optimize import brentq

    for vms, r34, b in [(100.0, 250.0, 1.3), (60.0, 150.0, 1.0), (140.0, 300.0, 2.2)]:
        rm = fp.rmax_from_r34_km(vms, r34, b)
        assert fp.symmetric_profile(r34, vms, rm, b) == pytest.approx(34.0, rel=1e-10)
        ref = brentq(lambda x, v=vms, r=r34, bb=b: fp.symmetric_profile(r, v, x, bb) - 34.0,
                     1e-6, r34 - 1e-9)
        assert rm == pytest.approx(ref, rel=1e-9)
    assert np.isnan(fp.rmax_from_r34_km(30.0, 100.0, 1.3))


def test_zero_translation_gives_a_symmetric_field():
    from landfall.geo import haversine_km

    rng = np.random.default_rng(0)
    lat_p, lon_p = 25 + rng.uniform(-2, 2, 200), -80 + rng.uniform(-2, 2, 200)
    w = fp.wind_speed(25.0, -80.0, 120.0, 30.0, 1.5, 0.0, 0.0, lat_p, lon_p)
    d = haversine_km(25.0, -80.0, lat_p, lon_p)
    np.testing.assert_allclose(w, fp.symmetric_profile(d, 120.0, 30.0, 1.5), rtol=1e-12)


def test_translation_asymmetry_right_and_left_of_motion():
    # Moving north at 36 km/h (19.44 kt). East of the centre (right of motion) at Rm the wind
    # is Vmax (Vms + c/2 with Vms = Vmax - c/2); west (left) it is Vmax - c.
    c_kt = 36.0 / 1.852
    lat_e, lon_e = offset_latlon(25.0, -80.0, 30.0, 0.0)
    lat_w, lon_w = offset_latlon(25.0, -80.0, -30.0, 0.0)
    right = fp.wind_speed(25.0, -80.0, 100.0, 30.0, 1.4, 0.0, 36.0, lat_e, lon_e)
    left = fp.wind_speed(25.0, -80.0, 100.0, 30.0, 1.4, 0.0, 36.0, lat_w, lon_w)
    assert right == pytest.approx(100.0, abs=0.05)
    assert left == pytest.approx(100.0 - c_kt, abs=0.05)


def test_inland_decay_by_hand_and_monotone():
    assert fp.inland_decay(100.0, 0.0) == pytest.approx(90.0)
    assert fp.inland_decay(100.0, 10.0) == pytest.approx(26.7 + 63.3 * np.exp(-0.95), rel=1e-14)
    v = fp.inland_decay(120.0, np.arange(0, 200))
    assert (np.diff(v) < 0).all() and v[-1] > 26.7 and v[-1] < 27.0
    assert fp.inland_decay(25.0, 30.0) == pytest.approx(22.5)


# Land east of 80W (20-30N).
GRID = LandGrid(20.0, -90.0, 0.1, 0.1, np.zeros((100, 200), dtype=np.uint8))
GRID.land[:, 100:] = 1


def _storms(n=3, seed=0):
    rng = np.random.default_rng(seed)
    node = np.array([0.0, 12, 24, 48, 72])
    lat = 25 + np.cumsum(rng.normal(0, 0.4, (n, 5)), axis=1)
    lon = -84 + np.cumsum(np.abs(rng.normal(1.2, 0.5, (n, 5))), axis=1)
    vmax = rng.uniform(60, 140, (n, 5))
    return fp.StormSet(node, lat, lon, vmax, rng.uniform(20, 60, n), rng.uniform(1.0, 2.2, n))


def test_decay_not_applied_before_landfall():
    s = _storms(5, seed=1)
    sea = fp.StormSet(s.node_h, s.lat, s.lon - 20.0, s.vmax_kt, s.rm_km, s.b)  # never on land
    lat_p, lon_p = np.array([25.0, 26.0]), np.array([-102.0, -101.0])
    np.testing.assert_array_equal(fp.max_wind_numpy(sea, lat_p, lon_p, GRID, decay=True),
                                  fp.max_wind_numpy(sea, lat_p, lon_p, GRID, decay=False))
    lat_q, lon_q = np.array([25.0, 25.0]), np.array([-83.0, -77.0])  # sea side, land side
    w_dec = fp.max_wind_numpy(s, lat_q, lon_q, GRID, decay=True)
    w_raw = fp.max_wind_numpy(s, lat_q, lon_q, GRID, decay=False)
    assert (w_dec <= w_raw + 1e-12).all() and (w_dec[:, 1] < w_raw[:, 1]).any()


def test_max_wind_equals_brute_force_over_substeps():
    s = _storms(2, seed=2)
    lat_p, lon_p = np.array([24.5, 26.0, 27.0]), np.array([-83.0, -80.5, -78.0])
    got = fp.max_wind_numpy(s, lat_p, lon_p, GRID, dt_h=1.0, decay=False)
    from landfall.geo import displacement_km

    for n in range(2):
        best = np.zeros(3)
        for t in np.arange(0, 72.0 + 1e-9, 1.0):
            k = min(np.searchsorted(s.node_h, t, side="right") - 1, len(s.node_h) - 2)
            f = (t - s.node_h[k]) / (s.node_h[k + 1] - s.node_h[k])
            la = s.lat[n, k] + (s.lat[n, k + 1] - s.lat[n, k]) * f
            lo = s.lon[n, k] + (s.lon[n, k + 1] - s.lon[n, k]) * f
            vm = s.vmax_kt[n, k] + (s.vmax_kt[n, k + 1] - s.vmax_kt[n, k]) * f
            e, nn = displacement_km(s.lat[n, k], s.lon[n, k], s.lat[n, k + 1], s.lon[n, k + 1])
            dur = s.node_h[k + 1] - s.node_h[k]
            w = fp.wind_speed(la, lo, vm, s.rm_km[n], s.b[n], e / dur, nn / dur, lat_p, lon_p)
            best = np.maximum(best, w)
        np.testing.assert_allclose(got[n], best, rtol=1e-13)


def test_exceedance_by_hand():
    mw = np.array([[10.0, 70.0], [50.0, 80.0], [64.0, 20.0], [100.0, 64.0]])
    np.testing.assert_array_equal(fp.exceedance(mw, [34, 64]), [[0.75, 0.75], [0.5, 0.75]])


# ---- C++ kernel against the NumPy reference --------------------------------------------------

@pytest.mark.parametrize("decay", [True, False])
def test_cpp_matches_numpy_synthetic(decay):
    cpp_module()
    s = _storms(50, seed=3)
    rng = np.random.default_rng(4)
    lat_p, lon_p = rng.uniform(22, 29, 40), rng.uniform(-86, -74, 40)
    ref = fp.max_wind_numpy(s, lat_p, lon_p, GRID, dt_h=0.5, decay=decay)
    got = fp.max_wind_cpp(s, lat_p, lon_p, GRID, dt_h=0.5, decay=decay)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)
    m = cpp_module()
    np.testing.assert_array_equal(m.exceedance(got, np.array([34.0, 64.0])),
                                  fp.exceedance(got, [34.0, 64.0]))


def test_cpp_matches_numpy_on_real_mask(land_grid):
    cpp_module()
    s = _storms(30, seed=5)
    rng = np.random.default_rng(6)
    lat_p, lon_p = rng.uniform(24, 31, 25), rng.uniform(-91, -79, 25)
    g = land_grid.coarsen(6)
    ref = fp.max_wind_numpy(s, lat_p, lon_p, g, dt_h=0.5)
    got = fp.max_wind_cpp(s, lat_p, lon_p, g, dt_h=0.5)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


def test_cpp_scalar_functions_match():
    m = cpp_module()
    for r in (0.0, 5.0, 30.0, 400.0):
        assert m.symmetric_profile(r, 90.0, 30.0, 1.4) == pytest.approx(
            float(fp.symmetric_profile(r, 90.0, 30.0, 1.4)), rel=1e-13, abs=1e-13)
    for h in (0.0, 3.0, 48.0):
        assert m.inland_decay(110.0, h) == pytest.approx(float(fp.inland_decay(110.0, h)),
                                                         rel=1e-14)
    assert m.wind_speed(25, -80, 100, 30, 1.4, 10, 20, 25.4, -79.6) == pytest.approx(
        float(fp.wind_speed(25, -80, 100, 30, 1.4, 10, 20, 25.4, -79.6)), rel=1e-12)


def test_cpp_rejects_bad_input():
    m = cpp_module()
    s = _storms(2)
    with pytest.raises(ValueError):
        m.max_wind(np.array([0.0, 0.0, 1, 2, 3]), s.lat, s.lon, s.vmax_kt, s.rm_km, s.b,
                   np.zeros(1), np.zeros(1), 0, 0, 1, 1, np.zeros((1, 1), np.uint8), 0.5, True)
    with pytest.raises(ValueError):
        m.max_wind(s.node_h, s.lat[:, :3], s.lon, s.vmax_kt, s.rm_km, s.b, np.zeros(1),
                   np.zeros(1), 0, 0, 1, 1, np.zeros((1, 1), np.uint8), 0.5, True)
