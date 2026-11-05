// Dependency-free tests of the footprint kernel; built and run by CTest, also under
// ASan/UBSan (LANDFALL_SANITIZE=ON). Hand-calculated values are documented next to each check.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
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


// The optimised kernel against the wave-1 reference kernel on random storms in four regions:
// anywhere, near the pole, across the date line, and with points at the storm centre.
void test_optimised_equals_reference() {
  struct Box {
    const char* name;
    double clat0, clat1, clon0, clon1, plat0, plat1, plon0, plon1;
  };
  const Box boxes[] = {
      {"random", -60, 60, -180, 180, -60, 60, -180, 180},
      {"near the pole", 86, 89, -180, 180, 87, 89.999, -180, 180},
      {"across the date line", 10, 30, 179, 179.8, 10, 30, -179.9, -179},
      {"near the storm centre", 24.9, 25.1, -80.1, -79.9, 24.9, 25.1, -80.1, -79.9},
  };
  std::vector<std::uint8_t> land(1, 0);
  const landfall::LandGridView g{0.0, 0.0, 1.0, 1.0, 1, 1, land.data()};
  std::mt19937_64 rng(42);
  auto unif = [&rng](double lo, double hi) {
    return std::uniform_real_distribution<double>(lo, hi)(rng);
  };
  const std::int64_t n = 8, k = 4, np = 40;
  const std::vector<double> node{0.0, 6.0, 12.0, 24.0};
  for (const Box& bx : boxes) {
    std::vector<double> la(n * k), lo(n * k), vm(n * k), rm(n), b(n), plat(np), plon(np);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n * k); ++i) {
      la[i] = unif(bx.clat0, bx.clat1);
      lo[i] = unif(bx.clon0, bx.clon1);
      vm[i] = unif(40, 160);
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
      rm[i] = unif(10, 80);
      b[i] = unif(1.0, 2.5);
    }
    for (std::size_t p = 0; p < static_cast<std::size_t>(np); ++p) {
      plat[p] = unif(bx.plat0, bx.plat1);
      plon[p] = unif(bx.plon0, bx.plon1);
    }
    plat[0] = la[0];  // a point exactly at the first centre
    plon[0] = lo[0];
    const landfall::StormArrays s{node.data(), k, la.data(), lo.data(), vm.data(),
                                  rm.data(),   b.data(), n};
    std::vector<double> fast(n * np), ref(n * np);
    const bool ok = landfall::max_wind(s, plat.data(), plon.data(), np, g, 0.5, true,
                                       fast.data()) &&
                    landfall::max_wind_reference(s, plat.data(), plon.data(), np, g, 0.5, true,
                                                 ref.data());
    double worst = 0.0;
    for (std::size_t i = 0; i < fast.size(); ++i) {
      worst = std::max(worst, std::fabs(fast[i] - ref[i]) / std::max(std::fabs(ref[i]), 1e-12));
    }
    if (!(ok && worst <= 1e-9)) std::printf("  %s: worst relative difference %.3g\n", bx.name,
                                            worst);
    check(ok && worst <= 1e-9, "optimised kernel equals the reference kernel within 1e-9");
  }
}

}  // namespace

int main() {
  test_profile();
  test_decay();
  test_wind_speed();
  test_max_wind_and_grid();
  test_optimised_equals_reference();
  if (failures == 0) std::printf("all footprint kernel tests passed\n");
  return failures == 0 ? 0 : 1;
}
