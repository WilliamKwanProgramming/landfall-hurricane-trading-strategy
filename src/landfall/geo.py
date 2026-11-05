"""Spherical geometry helpers and an offline land mask for the North Atlantic.

Conventions used everywhere in the package: latitude in degrees north, longitude in degrees east
(so the western hemisphere is negative), distances in kilometres, bearings in degrees clockwise
from north, times in hours.
"""

from __future__ import annotations

import hashlib
import pathlib
import zipfile
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

EARTH_RADIUS_KM = 6371.0
"""Mean Earth radius (km) used for every distance in the package."""

KM_PER_NMI = 1.852
"""Kilometres per nautical mile (exact by definition)."""

MS_PER_KT = 1852.0 / 3600.0
"""Metres per second per knot (exact by definition)."""


def haversine_km(lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike) -> NDArray:
    """Great-circle distance in km.

    d = 2 R asin( sqrt( sin^2(dphi/2) + cos(phi1) cos(phi2) sin^2(dlambda/2) ) ).
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike) -> NDArray:
    """Initial great-circle bearing from point 1 to point 2 (degrees clockwise from north).

    theta = atan2( sin(dlambda) cos(phi2), cos(phi1) sin(phi2) - sin(phi1) cos(phi2) cos(dlambda) ).
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360.0


def displacement_km(
    lat0: ArrayLike, lon0: ArrayLike, lat1: ArrayLike, lon1: ArrayLike
) -> tuple[NDArray, NDArray]:
    """East and north displacement (km) from point 0 to point 1 on the local equirectangular plane.

    east = R cos(phi0) dlambda, north = R dphi (angles in radians). `offset_latlon` is its exact
    inverse, so forecasts expressed as displacements convert back to positions without error.
    """
    lat0 = np.asarray(lat0, dtype=float)
    east = EARTH_RADIUS_KM * np.cos(np.radians(lat0)) * np.radians(
        np.asarray(lon1, dtype=float) - np.asarray(lon0, dtype=float))
    north = EARTH_RADIUS_KM * np.radians(np.asarray(lat1, dtype=float) - lat0)
    return east, north


def offset_latlon(
    lat0: ArrayLike, lon0: ArrayLike, east_km: ArrayLike, north_km: ArrayLike
) -> tuple[NDArray, NDArray]:
    """Inverse of `displacement_km`: the point reached from (lat0, lon0) by (east, north) km."""
    lat0 = np.asarray(lat0, dtype=float)
    lat1 = lat0 + np.degrees(np.asarray(north_km, dtype=float) / EARTH_RADIUS_KM)
    lon1 = np.asarray(lon0, dtype=float) + np.degrees(
        np.asarray(east_km, dtype=float) / (EARTH_RADIUS_KM * np.cos(np.radians(lat0))))
    return lat1, lon1


@dataclass(frozen=True)
class LandGrid:
    """A regular latitude/longitude grid of land (1) and sea (0) cells.

    Cell (i, j) covers lat in [lat0 + i*dlat, lat0 + (i+1)*dlat) and lon in
    [lon0 + j*dlon, lon0 + (j+1)*dlon). Points outside the grid are sea. Lookup is
    i = floor((lat - lat0) / dlat), j = floor((lon - lon0) / dlon); the C++ kernel uses the same
    rule on the same arrays, so both paths classify every point identically.
    """

    lat0: float
    lon0: float
    dlat: float
    dlon: float
    land: NDArray[np.uint8]

    def is_land(self, lat: ArrayLike, lon: ArrayLike) -> NDArray[np.bool_]:
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        i = np.floor((lat - self.lat0) / self.dlat)
        j = np.floor((lon - self.lon0) / self.dlon)
        n_i, n_j = self.land.shape
        ok = (i >= 0) & (i < n_i) & (j >= 0) & (j < n_j) & np.isfinite(i) & np.isfinite(j)
        out = np.zeros(np.broadcast(lat, lon).shape, dtype=bool)
        ii = np.where(ok, i, 0).astype(np.int64)
        jj = np.where(ok, j, 0).astype(np.int64)
        out[...] = ok & (self.land[ii, jj] == 1)
        return out

    def coarsen(self, factor: int) -> LandGrid:
        """Grid with cells `factor` times larger; a coarse cell is land if most fine cells are."""
        n_i = self.land.shape[0] // factor
        n_j = self.land.shape[1] // factor
        block = self.land[: n_i * factor, : n_j * factor].reshape(n_i, factor, n_j, factor)
        frac = block.mean(axis=(1, 3))
        return LandGrid(self.lat0, self.lon0, self.dlat * factor, self.dlon * factor,
                        (frac >= 0.5).astype(np.uint8))


# North Atlantic crop of the GLOBE-derived mask bundled with the `global-land-mask` package.
ATLANTIC_BOX = (0.0, 65.0, -110.0, 10.0)  # lat_min, lat_max, lon_min, lon_max (degrees)


def _mask_npz_path() -> pathlib.Path:
    import global_land_mask

    return pathlib.Path(global_land_mask.__file__).with_name("globe_combined_mask_compressed.npz")


def _read_npy_header(fh) -> tuple[tuple[int, ...], bool, np.dtype]:
    from numpy.lib import format as npf

    version = npf.read_magic(fh)
    if version == (1, 0):
        return npf.read_array_header_1_0(fh)
    return npf.read_array_header_2_0(fh)


def _crop_global_mask(box: tuple[float, float, float, float]) -> LandGrid:
    """Stream-decompress the bundled 1/120-degree mask and keep only the rows/columns of `box`.

    The bundled array `mask` is True for ocean, rows run from 90N southwards and columns from 180W
    eastwards (checked against the bundled `lat` and `lon` vectors). Reading it whole needs about
    1.8 GB of memory; streaming row by row keeps the peak near the size of the crop.
    """
    lat_min, lat_max, lon_min, lon_max = box
    path = _mask_npz_path()
    with zipfile.ZipFile(path) as zf:
        with zf.open("lat.npy") as fh:
            lat = np.lib.format.read_array(fh)
        with zf.open("lon.npy") as fh:
            lon = np.lib.format.read_array(fh)
        rows = np.nonzero((lat >= lat_min) & (lat < lat_max))[0]
        cols = np.nonzero((lon >= lon_min) & (lon < lon_max))[0]
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        with zf.open("mask.npy") as fh:
            shape, fortran, dtype = _read_npy_header(fh)
            if fortran or dtype != np.bool_ or shape != (lat.size, lon.size):
                raise ValueError(f"unexpected mask layout {shape} {dtype} fortran={fortran}")
            ncol = shape[1]
            fh.read(r0 * ncol)  # skip rows north of the box
            buf = np.frombuffer(fh.read((r1 - r0) * ncol), dtype=np.bool_).reshape(r1 - r0, ncol)
            ocean = buf[:, c0:c1]
    # Flip to ascending latitude so that row index grows northwards.
    land = np.logical_not(ocean)[::-1].astype(np.uint8)
    dlat = float(abs(lat[1] - lat[0]))
    dlon = float(lon[1] - lon[0])
    # The package maps a latitude to row int((lat - 90) / -dlat), so row r covers
    # (lat[r] - dlat, lat[r]]; the southern edge of the crop is lat[r1 - 1] - dlat.
    lat_south_edge = float(lat[r1 - 1]) - dlat
    return LandGrid(lat0=lat_south_edge, lon0=float(lon[c0]), dlat=dlat, dlon=dlon,
                    land=np.ascontiguousarray(land))


def _cache_dir() -> pathlib.Path:
    from landfall.data.paths import data_dir

    return data_dir() / "cache"


def atlantic_land_grid(coarsen: int = 1, cache: bool = True) -> LandGrid:
    """North Atlantic land mask (lat 0-65N, lon 110W-10E) from the `global-land-mask` package.

    Resolution 1/120 degree (about 0.9 km) before coarsening. Lakes count as land in the source
    data. The crop is cached (bit-packed) under data/cache/, keyed by the SHA-256 of the bundled
    file, so only the first call pays for the decompression.
    """
    src = _mask_npz_path()
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    cache_file = _cache_dir() / f"landmask_atlantic_{digest}.npz"
    grid = None
    if cache and cache_file.exists():
        z = np.load(cache_file)
        shape = tuple(int(v) for v in z["shape"])
        land = np.unpackbits(z["packed"])[: shape[0] * shape[1]].reshape(shape)
        grid = LandGrid(float(z["lat0"]), float(z["lon0"]), float(z["dlat"]), float(z["dlon"]),
                        land.astype(np.uint8))
    if grid is None:
        grid = _crop_global_mask(ATLANTIC_BOX)
        if cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_file, packed=np.packbits(grid.land),
                                shape=np.array(grid.land.shape), lat0=grid.lat0, lon0=grid.lon0,
                                dlat=grid.dlat, dlon=grid.dlon)
    return grid.coarsen(coarsen) if coarsen > 1 else grid
