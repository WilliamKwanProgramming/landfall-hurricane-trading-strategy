"""Parametric wind footprint: Holland (1980) profile, motion asymmetry and inland decay.

NumPy reference implementation; the C++ extension `landfall_cpp` implements the same kernel and
is tested against this module to 1e-9 relative on identical inputs. Northern hemisphere only.
Winds in knots, distances in km, times in hours.

Profile (Holland 1980, cyclostrophic form, as written in the PaHM model documentation):
    V(r) = Vm * sqrt( (Rm/r)^B * exp(1 - (Rm/r)^B) ),
so V(Rm) = Vm exactly. Coriolis is neglected (it lowers the gradient wind at Rm by about
r f / 2 ~ 1 kt near the core) and there is no inflow angle: the symmetric wind is purely
tangential and cyclonic.

B from the pressure deficit (Holland 1980, cyclostrophic balance at Rm): B = rho e Vg^2 / dp,
with rho = 1.15 kg m^-3, dp = (1013 hPa - pc) in Pa and Vg the gradient-level maximum wind,
taken as the surface Vmax / 0.9 (`SURFACE_FACTOR`, an assumption). B is clipped to [1, 2.5]
(the plausible range stated with the model); without a pressure B = 1.31, the mean of the 493
profiles fitted by Willoughby and Rahn (2004).

Radius of maximum wind: Willoughby, Darling and Rahn (2006), eq. 7a,
    Rm [km] = 46.4 exp(-0.0155 Vg + 0.0169 |lat|),  Vg in m/s (gradient level, as above);
or, when the 34-kt wind radii are known, the Rm for which the symmetric profile reaches 34 kt at
the mean of the non-zero 34-kt radii. That has a closed form: with q = (34 / Vm)^2 and
x = (Rm / r34)^B, x e^(1 - x) = q gives x = -W0(-q / e) (Lambert W, principal branch), so
Rm = r34 x^(1/B).

Motion asymmetry (Phadke et al. 2003, eq. 12, as implemented in the R package stormwindmodel):
the symmetric maximum is Vms = Vmax - 0.5 c (c the translation speed), and the translation
vector scaled by k(r) = Rm r / (Rm^2 + r^2) is added to the tangential wind vector. k(Rm) = 1/2,
so on the right of the motion at Rm the speed is again Vmax.

Inland decay (Kaplan and DeMaria 1995; constants from the NHC operational implementation,
routine decayo.f90 of the NCEP nhcg package, east/Gulf coast values): after the first landfall,
    V(t) = Vb + (R V0 - Vb) exp(-alpha (t - tL)),  R = 0.9, alpha = 0.095 / h, Vb = 26.7 kt,
with V0 the maximum wind at landfall. If R V0 <= Vb the wind is held at R V0 (the formula would
increase it). Decay continues for the rest of the track even if the centre returns to sea (a
simplification). Landfall is the first step whose centre is on land after a step at sea.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from landfall.geo import KM_PER_NMI, MS_PER_KT, LandGrid, bearing_deg, haversine_km

RHO_AIR = 1.15
P_ENV_HPA = 1013.0
SURFACE_FACTOR = 0.9
B_MIN, B_MAX, B_DEFAULT = 1.0, 2.5, 1.31
KD_R, KD_ALPHA, KD_VB = 0.9, 0.095, 26.7
KMH_PER_KT = KM_PER_NMI  # 1 kt = 1.852 km/h


def holland_b(vmax_kt: ArrayLike, pc_mb: ArrayLike) -> NDArray:
    """Holland B from the pressure deficit, clipped to [1, 2.5]; 1.31 when pc is unusable."""
    vg = np.asarray(vmax_kt, dtype=float) * MS_PER_KT / SURFACE_FACTOR
    dp = (P_ENV_HPA - np.asarray(pc_mb, dtype=float)) * 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        b = RHO_AIR * np.e * vg ** 2 / dp
    b = np.where(np.isfinite(b) & (dp > 0), np.clip(b, B_MIN, B_MAX), B_DEFAULT)
    return b


def rmax_willoughby_km(vmax_kt: ArrayLike, lat: ArrayLike) -> NDArray:
    """Willoughby, Darling and Rahn (2006) eq. 7a with Vg = Vmax / 0.9 in m/s."""
    vg = np.asarray(vmax_kt, dtype=float) * MS_PER_KT / SURFACE_FACTOR
    return 46.4 * np.exp(-0.0155 * vg + 0.0169 * np.abs(np.asarray(lat, dtype=float)))


def rmax_from_r34_km(vms_kt: float, r34_km: float, b: float) -> float:
    """Rm such that the symmetric Holland profile equals 34 kt at r34 (closed form, Lambert W).

    Requires vms_kt > 34 and r34_km > 0; returns NaN otherwise.
    """
    from scipy.special import lambertw

    if not (vms_kt > 34.0 and r34_km > 0):
        return float("nan")
    q = (34.0 / vms_kt) ** 2
    x = float(np.real(-lambertw(-q / np.e, 0)))
    return r34_km * x ** (1.0 / b)


def symmetric_profile(r_km: ArrayLike, vm_kt: ArrayLike, rm_km: ArrayLike, b: ArrayLike
                      ) -> NDArray:
    """Holland (1980) cyclostrophic profile V(r) = Vm sqrt((Rm/r)^B exp(1 - (Rm/r)^B))."""
    r = np.asarray(r_km, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = (np.asarray(rm_km, dtype=float) / r) ** np.asarray(b, dtype=float)
        v = np.asarray(vm_kt, dtype=float) * np.sqrt(x * np.exp(1.0 - x))
    return np.where(r > 0, np.nan_to_num(v, nan=0.0, posinf=0.0), 0.0)


def wind_speed(lat_c: ArrayLike, lon_c: ArrayLike, vmax_kt: ArrayLike, rm_km: ArrayLike,
               b: ArrayLike, ce_kmh: ArrayLike, cn_kmh: ArrayLike, lat_p: ArrayLike,
               lon_p: ArrayLike) -> NDArray:
    """Surface wind speed (kt) at points from storm centres, all arguments broadcast.

    (ce, cn) is the translation velocity (east, north) in km/h.
    """
    d = haversine_km(lat_c, lon_c, lat_p, lon_p)
    th = np.radians(bearing_deg(lat_c, lon_c, lat_p, lon_p))
    ce = np.asarray(ce_kmh, dtype=float) / KMH_PER_KT
    cn = np.asarray(cn_kmh, dtype=float) / KMH_PER_KT
    c = np.hypot(ce, cn)
    vms = np.maximum(np.asarray(vmax_kt, dtype=float) - 0.5 * c, 0.0)
    vs = symmetric_profile(d, vms, rm_km, b)
    rm = np.asarray(rm_km, dtype=float)
    k = np.where(d > 0, rm * d / (rm ** 2 + d ** 2), 0.0)
    # Cyclonic (counter-clockwise) tangential unit vector: (-cos th, sin th) in (east, north).
    ue = -vs * np.cos(th) + k * ce
    un = vs * np.sin(th) + k * cn
    return np.hypot(ue, un)


def inland_decay(v0_kt: ArrayLike, hours_since_landfall: ArrayLike) -> NDArray:
    """Kaplan-DeMaria (1995) decay of the maximum wind after landfall (see module docstring)."""
    v0 = np.asarray(v0_kt, dtype=float)
    t = np.asarray(hours_since_landfall, dtype=float)
    rv0 = KD_R * v0
    decayed = KD_VB + (rv0 - KD_VB) * np.exp(-KD_ALPHA * t)
    return np.where(rv0 > KD_VB, decayed, rv0)


@dataclass(frozen=True)
class StormSet:
    """N storms given at common node times (hours): positions and maximum wind at each node,
    and per-storm radius of maximum wind (km) and Holland B."""

    node_h: NDArray  # (K,)
    lat: NDArray  # (N, K)
    lon: NDArray  # (N, K)
    vmax_kt: NDArray  # (N, K)
    rm_km: NDArray  # (N,)
    b: NDArray  # (N,)


def substeps(node_h: NDArray, dt_h: float) -> tuple[NDArray, NDArray, NDArray]:
    """Sub-step times from node_h[0] to node_h[-1] every dt_h (inclusive), the index of the
    segment each falls in, and the fraction of the way through that segment."""
    n = int(np.floor((node_h[-1] - node_h[0]) / dt_h + 1e-9)) + 1
    t = node_h[0] + dt_h * np.arange(n)
    seg = np.clip(np.searchsorted(node_h, t, side="right") - 1, 0, len(node_h) - 2)
    frac = (t - node_h[seg]) / (node_h[seg + 1] - node_h[seg])
    return t, seg, frac


def max_wind_numpy(storms: StormSet, lat_p: ArrayLike, lon_p: ArrayLike, grid: LandGrid,
                   dt_h: float = 0.5, decay: bool = True) -> NDArray:
    """Maximum surface wind (kt) over the whole track at each point, for each storm: (N, P).

    Positions and maximum wind are linear in time between nodes; the translation velocity on
    each segment is the node-to-node displacement (equirectangular plane at the segment start)
    divided by its duration.
    """
    from landfall.geo import displacement_km

    lat_p = np.asarray(lat_p, dtype=float)
    lon_p = np.asarray(lon_p, dtype=float)
    t, seg, frac = substeps(storms.node_h, dt_h)
    a, b_ = storms.lat[:, seg], storms.lat[:, seg + 1]
    lat = a + (b_ - a) * frac
    lon = storms.lon[:, seg] + (storms.lon[:, seg + 1] - storms.lon[:, seg]) * frac
    vmx = storms.vmax_kt[:, seg] + (storms.vmax_kt[:, seg + 1] - storms.vmax_kt[:, seg]) * frac
    dur = storms.node_h[seg + 1] - storms.node_h[seg]
    ce, cn = displacement_km(storms.lat[:, seg], storms.lon[:, seg], storms.lat[:, seg + 1],
                             storms.lon[:, seg + 1])
    ce, cn = ce / dur, cn / dur
    if decay:
        land = grid.is_land(lat, lon)  # (N, S)
        entered = land[:, 1:] & ~land[:, :-1]
        has = entered.any(axis=1)
        first = np.where(has, entered.argmax(axis=1) + 1, lat.shape[1])
        steps = np.arange(lat.shape[1])[None, :]
        after = steps >= first[:, None]
        v0 = vmx[np.arange(len(first)), np.minimum(first, lat.shape[1] - 1)]
        tl = t[np.minimum(first, lat.shape[1] - 1)]
        vdec = inland_decay(v0[:, None], t[None, :] - tl[:, None])
        vmx = np.where(after, vdec, vmx)
    out = np.zeros((lat.shape[0], lat_p.size))
    for s in range(lat.shape[1]):
        w = wind_speed(lat[:, s, None], lon[:, s, None], vmx[:, s, None],
                       storms.rm_km[:, None], storms.b[:, None], ce[:, s, None], cn[:, s, None],
                       lat_p[None, :], lon_p[None, :])
        np.maximum(out, w, out=out)
    return out


def exceedance(maxwind: NDArray, thresholds_kt: ArrayLike) -> NDArray:
    """Fraction of storms whose maximum wind at each point is >= each threshold: (T, P)."""
    th = np.asarray(thresholds_kt, dtype=float)
    return (maxwind[None, :, :] >= th[:, None, None]).mean(axis=1)


def max_wind_cpp(storms: StormSet, lat_p: ArrayLike, lon_p: ArrayLike, grid: LandGrid,
                 dt_h: float = 0.5, decay: bool = True) -> NDArray:
    """Same as `max_wind_numpy`, computed by the C++ extension `landfall_cpp`.

    Build it with `cmake -S . -B build && cmake --build build`; the module is written to
    build/cpp/ (add that directory to sys.path or PYTHONPATH).
    """
    import landfall_cpp

    return landfall_cpp.max_wind(
        storms.node_h, storms.lat, storms.lon, storms.vmax_kt, storms.rm_km, storms.b,
        np.asarray(lat_p, dtype=float), np.asarray(lon_p, dtype=float), grid.lat0, grid.lon0,
        grid.dlat, grid.dlon, grid.land, dt_h, decay)
