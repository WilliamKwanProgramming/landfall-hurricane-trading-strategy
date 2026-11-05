#include "landfall/footprint.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace landfall {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kEarthRadiusKm = 6371.0;
constexpr double kKmhPerKt = 1.852;
constexpr double kKdR = 0.9;
constexpr double kKdAlpha = 0.095;
constexpr double kKdVb = 26.7;

double radians(double deg) { return deg * (kPi / 180.0); }

double haversine_km(double lat1, double lon1, double lat2, double lon2) {
  const double p1 = radians(lat1);
  const double p2 = radians(lat2);
  const double dp = p2 - p1;
  const double dl = radians(lon2 - lon1);
  const double s1 = std::sin(dp / 2);
  const double s2 = std::sin(dl / 2);
  double a = s1 * s1 + std::cos(p1) * std::cos(p2) * s2 * s2;
  a = std::min(1.0, std::max(0.0, a));
  return 2.0 * kEarthRadiusKm * std::asin(std::sqrt(a));
}

}  // namespace

bool LandGridView::is_land(double lat, double lon) const {
  const double fi = std::floor((lat - lat0) / dlat);
  const double fj = std::floor((lon - lon0) / dlon);
  if (!(fi >= 0.0 && fj >= 0.0 && fi < static_cast<double>(n_lat) &&
        fj < static_cast<double>(n_lon))) {
    return false;  // outside the grid (or NaN): sea
  }
  const auto i = static_cast<std::int64_t>(fi);
  const auto j = static_cast<std::int64_t>(fj);
  return land[static_cast<std::size_t>(i * n_lon + j)] == 1;
}

double symmetric_profile(double r_km, double vm_kt, double rm_km, double b) {
  if (!(r_km > 0.0)) return 0.0;
  const double x = std::pow(rm_km / r_km, b);
  const double v = vm_kt * std::sqrt(x * std::exp(1.0 - x));
  return std::isfinite(v) ? v : 0.0;
}

double inland_decay(double v0_kt, double hours_since_landfall) {
  const double rv0 = kKdR * v0_kt;
  if (!(rv0 > kKdVb)) return rv0;
  return kKdVb + (rv0 - kKdVb) * std::exp(-kKdAlpha * hours_since_landfall);
}

double wind_speed(double lat_c, double lon_c, double vmax_kt, double rm_km, double b,
                  double ce_kmh, double cn_kmh, double lat_p, double lon_p) {
  const double d = haversine_km(lat_c, lon_c, lat_p, lon_p);
  // Bearing from the centre to the point; only its sine and cosine are needed.
  const double p1 = radians(lat_c);
  const double p2 = radians(lat_p);
  const double dl = radians(lon_p - lon_c);
  const double y = std::sin(dl) * std::cos(p2);
  const double x = std::cos(p1) * std::sin(p2) - std::sin(p1) * std::cos(p2) * std::cos(dl);
  const double h = std::hypot(x, y);
  const double cos_th = h > 0.0 ? x / h : 1.0;
  const double sin_th = h > 0.0 ? y / h : 0.0;
  const double ce = ce_kmh / kKmhPerKt;
  const double cn = cn_kmh / kKmhPerKt;
  const double c = std::hypot(ce, cn);
  const double vms = std::max(vmax_kt - 0.5 * c, 0.0);
  const double vs = symmetric_profile(d, vms, rm_km, b);
  const double k = d > 0.0 ? rm_km * d / (rm_km * rm_km + d * d) : 0.0;
  const double ue = -vs * cos_th + k * ce;
  const double un = vs * sin_th + k * cn;
  return std::hypot(ue, un);
}

namespace {

// Sub-step times and the interpolated track of one storm; shared by both kernels so that they
// differ only in the (step, point) loop. Identical to footprint.substeps() and the
// interpolation, translation velocity and decay of max_wind_numpy().
struct Track {
  std::vector<double> t, frac, lat, lon, vmx, ce, cn;
  std::vector<std::int64_t> seg;
};

bool make_steps(const StormArrays& s, double dt_h, std::int64_t n_points, Track& tr) {
  const std::int64_t K = s.n_nodes;
  if (K < 2 || !(dt_h > 0.0) || n_points < 0 || s.n_storms < 0) return false;
  for (std::int64_t k = 1; k < K; ++k) {
    if (!(s.node_h[k] > s.node_h[k - 1])) return false;
  }
  const double span = s.node_h[K - 1] - s.node_h[0];
  const auto n_steps = static_cast<std::int64_t>(std::floor(span / dt_h + 1e-9)) + 1;
  const auto nu = static_cast<std::size_t>(n_steps);
  tr.t.resize(nu);
  tr.seg.resize(nu);
  tr.frac.resize(nu);
  for (auto* v : {&tr.lat, &tr.lon, &tr.vmx, &tr.ce, &tr.cn}) v->resize(nu);
  for (std::size_t i = 0; i < nu; ++i) {
    tr.t[i] = s.node_h[0] + dt_h * static_cast<double>(i);
    const double* ub = std::upper_bound(s.node_h, s.node_h + K, tr.t[i]);
    std::int64_t g = static_cast<std::int64_t>(ub - s.node_h) - 1;
    g = std::min(std::max(g, std::int64_t{0}), K - 2);
    tr.seg[i] = g;
    tr.frac[i] = (tr.t[i] - s.node_h[g]) / (s.node_h[g + 1] - s.node_h[g]);
  }
  return true;
}

void fill_track(const StormArrays& s, std::int64_t n, const LandGridView& grid, bool decay,
                Track& tr) {
  const std::int64_t K = s.n_nodes;
  const double* la = s.lat + n * K;
  const double* lo = s.lon + n * K;
  const double* vm = s.vmax + n * K;
  const std::size_t n_steps = tr.t.size();
  for (std::size_t i = 0; i < n_steps; ++i) {
    const std::int64_t g = tr.seg[i];
    const double f = tr.frac[i];
    tr.lat[i] = la[g] + (la[g + 1] - la[g]) * f;
    tr.lon[i] = lo[g] + (lo[g + 1] - lo[g]) * f;
    tr.vmx[i] = vm[g] + (vm[g + 1] - vm[g]) * f;
    const double dur = s.node_h[g + 1] - s.node_h[g];
    const double east = kEarthRadiusKm * std::cos(radians(la[g])) * radians(lo[g + 1] - lo[g]);
    const double north = kEarthRadiusKm * radians(la[g + 1] - la[g]);
    tr.ce[i] = east / dur;
    tr.cn[i] = north / dur;
  }
  if (!decay) return;
  std::size_t first = n_steps;
  bool prev = grid.is_land(tr.lat[0], tr.lon[0]);
  for (std::size_t i = 1; i < n_steps; ++i) {
    const bool cur = grid.is_land(tr.lat[i], tr.lon[i]);
    if (cur && !prev) {
      first = i;
      break;
    }
    prev = cur;
  }
  if (first < n_steps) {
    const double v0 = tr.vmx[first];
    const double tl = tr.t[first];
    for (std::size_t i = first; i < n_steps; ++i) tr.vmx[i] = inland_decay(v0, tr.t[i] - tl);
  }
}

}  // namespace

bool max_wind_reference(const StormArrays& s, const double* lat_p, const double* lon_p,
                        std::int64_t n_points, const LandGridView& grid, double dt_h, bool decay,
                        double* out) {
  Track tr;
  if (!make_steps(s, dt_h, n_points, tr)) return false;
  const std::size_t n_steps = tr.t.size();
  for (std::int64_t n = 0; n < s.n_storms; ++n) {
    fill_track(s, n, grid, decay, tr);
    double* row = out + n * n_points;
    for (std::int64_t p = 0; p < n_points; ++p) row[p] = 0.0;
    for (std::size_t i = 0; i < n_steps; ++i) {
      for (std::int64_t p = 0; p < n_points; ++p) {
        const double w = wind_speed(tr.lat[i], tr.lon[i], tr.vmx[i], s.rm[n], s.b[n], tr.ce[i],
                                    tr.cn[i], lat_p[p], lon_p[p]);
        if (w > row[p]) row[p] = w;
      }
    }
  }
  return true;
}

bool max_wind(const StormArrays& s, const double* lat_p, const double* lon_p,
              std::int64_t n_points, const LandGridView& grid, double dt_h, bool decay,
              double* out) {
  Track tr;
  if (!make_steps(s, dt_h, n_points, tr)) return false;
  const std::size_t n_steps = tr.t.size();
  const auto np = static_cast<std::size_t>(n_points);

  // Per-point terms, computed once: sine and cosine of the latitude phi, of phi / 2 and of
  // lambda / 2 (lambda the longitude in radians).
  std::vector<double> sp(np), cp(np), shp(np), chp(np), slp(np), clp(np);
  for (std::size_t p = 0; p < np; ++p) {
    const double phi = radians(lat_p[p]);
    const double lam = radians(lon_p[p]);
    sp[p] = std::sin(phi);
    cp[p] = std::cos(phi);
    shp[p] = std::sin(0.5 * phi);
    chp[p] = std::cos(0.5 * phi);
    slp[p] = std::sin(0.5 * lam);
    clp[p] = std::cos(0.5 * lam);
  }

  for (std::int64_t n = 0; n < s.n_storms; ++n) {
    fill_track(s, n, grid, decay, tr);
    const double rm = s.rm[n];
    const double rm2 = rm * rm;
    const double b = s.b[n];
    double* row = out + n * n_points;
    for (std::size_t p = 0; p < np; ++p) row[p] = 0.0;
    for (std::size_t i = 0; i < n_steps; ++i) {
      // Per-(storm, step) terms, hoisted out of the point loop.
      const double phi = radians(tr.lat[i]);
      const double lam = radians(tr.lon[i]);
      const double sc = std::sin(phi);
      const double cc = std::cos(phi);
      const double shc = std::sin(0.5 * phi);
      const double chc = std::cos(0.5 * phi);
      const double slc = std::sin(0.5 * lam);
      const double clc = std::cos(0.5 * lam);
      const double ce = tr.ce[i] / kKmhPerKt;
      const double cn = tr.cn[i] / kKmhPerKt;
      const double vms = std::max(tr.vmx[i] - 0.5 * std::hypot(ce, cn), 0.0);
      for (std::size_t p = 0; p < np; ++p) {
        // Half-angle differences: sin((a - b) / 2) = sin(a/2) cos(b/2) - cos(a/2) sin(b/2) and
        // cos((a - b) / 2) = cos(a/2) cos(b/2) + sin(a/2) sin(b/2); then sin(dl) = 2 s c and
        // cos(dl) = 1 - 2 s^2. Valid for any longitude difference (no wrapping needed).
        const double s_dphi = shp[p] * chc - chp[p] * shc;
        const double s_dl = slp[p] * clc - clp[p] * slc;
        const double c_dl = clp[p] * clc + slp[p] * slc;
        // Haversine distance (same formula as haversine_km).
        double a = s_dphi * s_dphi + cc * cp[p] * s_dl * s_dl;
        a = std::min(1.0, std::max(0.0, a));
        const double d = 2.0 * kEarthRadiusKm * std::asin(std::sqrt(a));
        // Bearing from the centre to the point (same formula as wind_speed). |x| <= 2 and
        // |y| <= 1, so x^2 + y^2 cannot overflow and sqrt replaces hypot; the squares underflow
        // only within about 1e-148 km of the centre, where the profile is 0 and the direction
        // does not affect the result.
        const double y = (2.0 * s_dl * c_dl) * cp[p];
        const double x = cc * sp[p] - sc * cp[p] * (1.0 - 2.0 * s_dl * s_dl);
        const double h = std::sqrt(x * x + y * y);
        const double cos_th = h > 0.0 ? x / h : 1.0;
        const double sin_th = h > 0.0 ? y / h : 0.0;
        const double vs = symmetric_profile(d, vms, rm, b);
        const double k = d > 0.0 ? rm * d / (rm2 + d * d) : 0.0;
        const double ue = -vs * cos_th + k * ce;
        const double un = vs * sin_th + k * cn;
        // Wind speeds are bounded by physical inputs, but the inputs are not checked: fall back
        // to hypot if ue^2 + un^2 overflows (|ue| or |un| above about 1e154), so results never
        // differ from max_wind_reference for that reason.
        double w = std::sqrt(ue * ue + un * un);
        if (!std::isfinite(w)) w = std::hypot(ue, un);
        if (w > row[p]) row[p] = w;
      }
    }
  }
  return true;
}

void exceedance(const double* maxwind, std::int64_t n_storms, std::int64_t n_points,
                const double* thresholds, std::int64_t n_thresholds, double* out) {
  for (std::int64_t k = 0; k < n_thresholds; ++k) {
    for (std::int64_t p = 0; p < n_points; ++p) {
      std::int64_t hits = 0;
      for (std::int64_t n = 0; n < n_storms; ++n) {
        if (maxwind[n * n_points + p] >= thresholds[k]) ++hits;
      }
      out[k * n_points + p] =
          n_storms > 0 ? static_cast<double>(hits) / static_cast<double>(n_storms) : 0.0;
    }
  }
}

}  // namespace landfall
