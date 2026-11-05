// Wind-footprint Monte Carlo kernel: the C++ twin of landfall/footprint.py (max_wind_numpy).
//
// Every formula, constant and evaluation order follows the NumPy reference so that both paths
// agree to 1e-9 relative on identical inputs (tested from Python). See the Python module
// docstring for the sources of the formulas. Units: degrees, km, hours, knots.
#pragma once

#include <cstddef>
#include <cstdint>

namespace landfall {

// Regular land/sea grid. Cell (i, j) covers lat in [lat0 + i dlat, lat0 + (i + 1) dlat) and
// lon in [lon0 + j dlon, lon0 + (j + 1) dlon); points outside the grid are sea.
struct LandGridView {
  double lat0;
  double lon0;
  double dlat;
  double dlon;
  std::int64_t n_lat;
  std::int64_t n_lon;
  const std::uint8_t* land;  // row-major, n_lat * n_lon

  bool is_land(double lat, double lon) const;
};

// Holland (1980) cyclostrophic profile, Vm sqrt((Rm/r)^B exp(1 - (Rm/r)^B)); 0 for r <= 0.
double symmetric_profile(double r_km, double vm_kt, double rm_km, double b);

// Kaplan-DeMaria (1995) inland decay with the NHC operational east/Gulf coast constants.
double inland_decay(double v0_kt, double hours_since_landfall);

// Surface wind speed (kt) at one point from one storm centre; (ce, cn) translation in km/h.
double wind_speed(double lat_c, double lon_c, double vmax_kt, double rm_km, double b,
                  double ce_kmh, double cn_kmh, double lat_p, double lon_p);

// Inputs of max_wind. Arrays are row-major: lat, lon, vmax are (n_storms, n_nodes); rm and b are
// (n_storms); lat_p and lon_p are (n_points).
struct StormArrays {
  const double* node_h;
  std::int64_t n_nodes;
  const double* lat;
  const double* lon;
  const double* vmax;
  const double* rm;
  const double* b;
  std::int64_t n_storms;
};

// Maximum surface wind over each storm's track at each point; writes out[s * n_points + p].
// Returns false (and writes nothing) if the inputs are inconsistent (fewer than two nodes,
// non-increasing node times, dt_h <= 0).
//
// Optimised kernel: the sines and cosines that depend only on a point are computed once per
// point, those that depend only on the storm position once per (storm, step), and the
// trigonometric functions of the longitude and latitude differences come from half-angle
// identities, so the innermost loop calls only asin, pow, exp and sqrt. Same formulas as
// max_wind_reference; results agree with it and with NumPy within 1e-9 relative (tested; the
// largest difference seen, 2e-12, is for points a few km from the centre).
bool max_wind(const StormArrays& storms, const double* lat_p, const double* lon_p,
              std::int64_t n_points, const LandGridView& grid, double dt_h, bool decay,
              double* out);

// The first kernel, kept unchanged as a reference for tests and the benchmark: it
// calls wind_speed() for every (storm, step, point), i.e. about fifteen libm functions each.
bool max_wind_reference(const StormArrays& storms, const double* lat_p, const double* lon_p,
                        std::int64_t n_points, const LandGridView& grid, double dt_h, bool decay,
                        double* out);

// Fraction of storms with max wind >= threshold: out[t * n_points + p].
void exceedance(const double* maxwind, std::int64_t n_storms, std::int64_t n_points,
                const double* thresholds, std::int64_t n_thresholds, double* out);

}  // namespace landfall
