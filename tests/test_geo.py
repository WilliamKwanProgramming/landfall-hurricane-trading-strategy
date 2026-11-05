import numpy as np
import pytest

from landfall import geo


def test_haversine_known_values():
    # A quarter of a meridian and one degree of longitude at the equator.
    assert geo.haversine_km(0, 0, 90, 0) == pytest.approx(np.pi / 2 * geo.EARTH_RADIUS_KM)
    assert geo.haversine_km(0, 0, 0, 1) == pytest.approx(geo.EARTH_RADIUS_KM * np.pi / 180)
    assert geo.haversine_km(25, -80, 25, -80) == 0.0


def test_bearing_cardinal_directions():
    assert geo.bearing_deg(0, 0, 1, 0) == pytest.approx(0.0)
    assert geo.bearing_deg(0, 0, 0, 1) == pytest.approx(90.0)
    assert geo.bearing_deg(0, 0, -1, 0) == pytest.approx(180.0)
    assert geo.bearing_deg(0, 0, 0, -1) == pytest.approx(270.0)


def test_displacement_round_trip():
    rng = np.random.default_rng(1)
    lat0 = rng.uniform(5, 50, 100)
    lon0 = rng.uniform(-100, -20, 100)
    e, n = rng.normal(0, 500, 100), rng.normal(0, 500, 100)
    lat1, lon1 = geo.offset_latlon(lat0, lon0, e, n)
    e2, n2 = geo.displacement_km(lat0, lon0, lat1, lon1)
    np.testing.assert_allclose(e2, e, atol=1e-9)
    np.testing.assert_allclose(n2, n, atol=1e-9)


def test_displacement_close_to_great_circle_for_short_hops():
    e, n = geo.displacement_km(25.0, -80.0, 25.5, -80.5)
    d = geo.haversine_km(25.0, -80.0, 25.5, -80.5)
    assert np.hypot(e, n) == pytest.approx(d, rel=5e-3)


def test_land_grid_lookup_rule():
    g = geo.LandGrid(10.0, -50.0, 1.0, 1.0, np.array([[0, 1], [1, 0]], dtype=np.uint8))
    assert g.is_land(10.5, -48.5) and g.is_land(11.0, -50.0)
    assert not g.is_land(10.5, -49.5) and not g.is_land(11.99, -48.01)
    assert not g.is_land(9.99, -49.5) and not g.is_land(12.0, -50.0)  # outside: sea
    assert not g.is_land(np.nan, -49.5)


def test_atlantic_mask_matches_package_lookup(land_grid):
    from global_land_mask import globe

    rng = np.random.default_rng(7)
    lat = rng.uniform(5, 60, 20000)
    lon = rng.uniform(-105, 5, 20000)
    agree = np.mean(land_grid.is_land(lat, lon) == globe.is_land(lat, lon))
    assert agree > 0.999


@pytest.mark.parametrize("lat,lon,land", [
    (25.77, -80.19, True),    # Miami
    (29.95, -90.07, True),    # New Orleans
    (25.0, -85.0, False),     # Gulf of Mexico
    (30.0, -60.0, False),     # open Atlantic
    (40.71, -74.0, True),     # Manhattan
])
def test_atlantic_mask_known_points(land_grid, lat, lon, land):
    assert bool(land_grid.is_land(lat, lon)) is land
