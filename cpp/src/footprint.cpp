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

bool max_wind(const StormArrays& s, const double* lat_p, const double* lon_p,
              std::int64_t n_points, const LandGridView& grid, double dt_h, bool decay,
              double* out) {
  const std::int64_t K = s.n_nodes;
  if (K < 2 || !(dt_h > 0.0) || n_points < 0 || s.n_storms < 0) return false;
  for (std::int64_t k = 1; k < K; ++k) {
    if (!(s.node_h[k] > s.node_h[k - 1])) return false;
  }
  // Sub-steps, identical to footprint.substeps().
  const double span = s.node_h[K - 1] - s.node_h[0];
  const auto n_steps = static_cast<std::int64_t>(std::floor(span / dt_h + 1e-9)) + 1;
  std::vector<double> t(static_cast<std::size_t>(n_steps));
  std::vector<std::int64_t> seg(static_cast<std::size_t>(n_steps));
  std::vector<double> frac(static_cast<std::size_t>(n_steps));
  for (std::int64_t i = 0; i < n_steps; ++i) {
    const auto iu = static_cast<std::size_t>(i);
    t[iu] = s.node_h[0] + dt_h * static_cast<double>(i);
    const double* ub = std::upper_bound(s.node_h, s.node_h + K, t[iu]);
    std::int64_t g = static_cast<std::int64_t>(ub - s.node_h) - 1;
    g = std::min(std::max(g, std::int64_t{0}), K - 2);
    seg[iu] = g;
    frac[iu] = (t[iu] - s.node_h[g]) / (s.node_h[g + 1] - s.node_h[g]);
  }

  std::vector<double> lat(static_cast<std::size_t>(n_steps));
  std::vector<double> lon(static_cast<std::size_t>(n_steps));
  std::vector<double> vmx(static_cast<std::size_t>(n_steps));
  std::vector<double> ce(static_cast<std::size_t>(n_steps));
  std::vector<double> cn(static_cast<std::size_t>(n_steps));
  for (std::int64_t n = 0; n < s.n_storms; ++n) {
    const double* la = s.lat + n * K;
    const double* lo = s.lon + n * K;
    const double* vm = s.vmax + n * K;
    for (std::int64_t i = 0; i < n_steps; ++i) {
      const auto iu = static_cast<std::size_t>(i);
      const std::int64_t g = seg[iu];
      const double f = frac[iu];
      lat[iu] = la[g] + (la[g + 1] - la[g]) * f;
      lon[iu] = lo[g] + (lo[g + 1] - lo[g]) * f;
      vmx[iu] = vm[g] + (vm[g + 1] - vm[g]) * f;
      const double dur = s.node_h[g + 1] - s.node_h[g];
      const double east = kEarthRadiusKm * std::cos(radians(la[g])) * radians(lo[g + 1] - lo[g]);
      const double north = kEarthRadiusKm * radians(la[g + 1] - la[g]);
      ce[iu] = east / dur;
      cn[iu] = north / dur;
    }
    if (decay) {
      std::int64_t first = n_steps;
      bool prev = grid.is_land(lat[0], lon[0]);
      for (std::int64_t i = 1; i < n_steps; ++i) {
        const auto iu = static_cast<std::size_t>(i);
        const bool cur = grid.is_land(lat[iu], lon[iu]);
        if (cur && !prev) {
          first = i;
          break;
        }
        prev = cur;
      }
      if (first < n_steps) {
        const auto fu = static_cast<std::size_t>(first);
        const double v0 = vmx[fu];
        const double tl = t[fu];
        for (std::int64_t i = first; i < n_steps; ++i) {
          const auto iu = static_cast<std::size_t>(i);
          vmx[iu] = inland_decay(v0, t[iu] - tl);
        }
      }
    }
    double* row = out + n * n_points;
    for (std::int64_t p = 0; p < n_points; ++p) row[p] = 0.0;
    for (std::int64_t i = 0; i < n_steps; ++i) {
      const auto iu = static_cast<std::size_t>(i);
      for (std::int64_t p = 0; p < n_points; ++p) {
        const double w = wind_speed(lat[iu], lon[iu], vmx[iu], s.rm[n], s.b[n], ce[iu], cn[iu],
                                    lat_p[p], lon_p[p]);
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
