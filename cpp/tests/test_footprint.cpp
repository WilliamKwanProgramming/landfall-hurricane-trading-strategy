// Dependency-free tests of the footprint kernel; built and run by CTest, also under
// ASan/UBSan (LANDFALL_SANITIZE=ON). Hand-calculated values are documented next to each check.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "landfall/footprint.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

bool close(double a, double b, double rel) {
  return std::fabs(a - b) <= rel * std::max(1.0, std::fabs(b));
}

void test_profile() {
  using landfall::symmetric_profile;
  check(symmetric_profile(30.0, 100.0, 30.0, 1.5) == 100.0, "V(Rm) == Vm exactly");
  check(symmetric_profile(0.0, 100.0, 30.0, 1.5) == 0.0, "V(0) == 0");
  check(symmetric_profile(-1.0, 100.0, 30.0, 1.5) == 0.0, "V(r<0) == 0");
  check(symmetric_profile(1e-300, 100.0, 30.0, 1.5) == 0.0, "V(r->0) == 0 without NaN");
  // r = 2 Rm, B = 1: x = 0.5, V = 100 sqrt(0.5 e^0.5) = 90.8560...
  check(close(symmetric_profile(60.0, 100.0, 30.0, 1.0), 100.0 * std::sqrt(0.5 * std::exp(0.5)),
              1e-14),
        "V(2 Rm) by hand");
  double best = 0.0, at = 0.0;
  for (int i = 1; i < 2000; ++i) {
    const double r = 0.1 * i;
    const double v = symmetric_profile(r, 80.0, 40.0, 1.3);
    if (v > best) {
      best = v;
      at = r;
    }
  }
  check(std::fabs(at - 40.0) < 0.11 && best <= 80.0, "maximum at Rm");
}

void test_decay() {
  using landfall::inland_decay;
  check(close(inland_decay(100.0, 0.0), 90.0, 1e-15), "decay(0) = 0.9 V0");
  // 26.7 + (90 - 26.7) exp(-0.95) = 51.1806...
  check(close(inland_decay(100.0, 10.0), 26.7 + 63.3 * std::exp(-0.95), 1e-14), "decay(10 h)");
  double prev = 1e9;
  bool mono = true;
  for (int h = 0; h <= 200; ++h) {
    const double v = inland_decay(120.0, h);
    mono = mono && v < prev && v > 26.7;
    prev = v;
  }
  check(mono, "decay strictly decreasing towards 26.7 kt");
  check(inland_decay(25.0, 30.0) == 0.9 * 25.0, "weak storm held at R V0");
}

void test_wind_speed() {
  using landfall::wind_speed;
  // A point due east of the centre at about Rm, no motion: Vmax.
  const double lat = 25.0, lon = -80.0;
  const double dlon_rm = 30.0 / (6371.0 * std::cos(lat * 3.14159265358979 / 180.0)) * 180.0 /
                         3.14159265358979;
  const double w0 = wind_speed(lat, lon, 100.0, 30.0, 1.4, 0.0, 0.0, lat, lon + dlon_rm);
  check(std::fabs(w0 - 100.0) < 0.01, "about Vmax at about Rm, no motion");
  // Zero motion: the field depends on distance only (north vs south at equal latitude offset
  // differ in distance slightly, so compare east and west, which are equidistant).
  const double we = wind_speed(lat, lon, 90.0, 40.0, 1.2, 0.0, 0.0, lat, lon + 0.5);
  const double ww = wind_speed(lat, lon, 90.0, 40.0, 1.2, 0.0, 0.0, lat, lon - 0.5);
  check(close(we, ww, 1e-12), "symmetric without motion");
  // Northward motion at 20 km/h: the right-hand side (east) is stronger than the left.
  const double re = wind_speed(lat, lon, 90.0, 40.0, 1.2, 0.0, 20.0, lat, lon + 0.5);
  const double rw = wind_speed(lat, lon, 90.0, 40.0, 1.2, 0.0, 20.0, lat, lon - 0.5);
  check(re > rw, "right side stronger with motion");
  check(wind_speed(lat, lon, 90.0, 40.0, 1.2, 5.0, 5.0, lat, lon) ==
            0.0,
        "zero at the centre");
}

void test_max_wind_and_grid() {
  // Land east of 80W; storm moving east along 25N crosses at about t = 12 h.
  const std::int64_t n_lat = 100, n_lon = 200;
  std::vector<std::uint8_t> land(static_cast<std::size_t>(n_lat * n_lon), 0);
  for (std::int64_t i = 0; i < n_lat; ++i)
    for (std::int64_t j = 100; j < n_lon; ++j) land[static_cast<std::size_t>(i * n_lon + j)] = 1;
  landfall::LandGridView g{20.0, -90.0, 0.1, 0.1, n_lat, n_lon, land.data()};
  check(g.is_land(25.0, -79.95) && !g.is_land(25.0, -80.05), "grid lookup");
  check(!g.is_land(19.99, -79.0) && !g.is_land(25.0, -69.99) && !g.is_land(NAN, -79.0),
        "outside or NaN is sea");

  const std::vector<double> node{0.0, 12.0, 24.0};
  const std::vector<double> la{25.0, 25.0, 25.0, 25.0, 25.0, 25.0};
  const std::vector<double> lo{-82.0, -80.0, -78.0, -82.0, -80.0, -78.0};
  const std::vector<double> vm{100.0, 100.0, 100.0, 100.0, 100.0, 100.0};
  const std::vector<double> rm{30.0, 30.0};
  const std::vector<double> b{1.3, 1.3};
  landfall::StormArrays s{node.data(), 3, la.data(), lo.data(), vm.data(), rm.data(), b.data(),
                          2};
  const std::vector<double> plat{25.3, 25.3};
  const std::vector<double> plon{-81.5, -78.5};  // one point before, one after landfall
  std::vector<double> with(4), without(4);
  check(landfall::max_wind(s, plat.data(), plon.data(), 2, g, 0.5, true, with.data()),
        "max_wind ok");
  check(landfall::max_wind(s, plat.data(), plon.data(), 2, g, 0.5, false, without.data()),
        "max_wind ok without decay");
  check(with[0] == without[0], "no decay effect before landfall");
  check(with[1] < without[1], "decay lowers winds inland");
  check(with[0] == with[2] && with[1] == with[3], "identical storms give identical rows");
  std::vector<double> bad(4);
  const std::vector<double> node_bad{0.0, 0.0, 24.0};
  landfall::StormArrays sb = s;
  sb.node_h = node_bad.data();
  check(!landfall::max_wind(sb, plat.data(), plon.data(), 2, g, 0.5, true, bad.data()),
        "non-increasing node times rejected");
  check(!landfall::max_wind(s, plat.data(), plon.data(), 2, g, 0.0, true, bad.data()),
        "dt <= 0 rejected");

  std::vector<double> ex(4);
  const std::vector<double> th{50.0, 1000.0};
  landfall::exceedance(with.data(), 2, 2, th.data(), 2, ex.data());
  check(ex[0] == 1.0 && ex[2] == 0.0, "exceedance fractions");
}

}  // namespace

int main() {
  test_profile();
  test_decay();
  test_wind_speed();
  test_max_wind_and_grid();
  if (failures == 0) std::printf("all footprint kernel tests passed\n");
  return failures == 0 ? 0 : 1;
}
