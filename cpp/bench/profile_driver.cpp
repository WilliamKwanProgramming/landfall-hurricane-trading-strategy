// Driver for experiments/profile_kernel.py: runs one footprint kernel once on synthetic storms
// (the same distributions as experiments/benchmark_footprint.py, other random numbers) so that
// callgrind can count the instructions per wind evaluation.
//   profile_driver <n_storms> <reference|optimised>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "landfall/footprint.hpp"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <n_storms> <reference|optimised>\n", argv[0]);
    return 2;
  }
  const std::int64_t n = std::atoll(argv[1]);
  const bool reference = std::strcmp(argv[2], "reference") == 0;
  const std::int64_t k = 7, np = 50;
  std::mt19937_64 rng(20261105);
  std::normal_distribution<double> dlat(0.5, 0.4), dlon(0.3, 0.6);
  std::uniform_real_distribution<double> uv(60, 140), urm(20, 60), ub(1.0, 2.2), ula(26, 31),
      ulo(-97, -82);
  const std::vector<double> node{0, 12, 24, 48, 72, 96, 120};
  const auto nk = static_cast<std::size_t>(n * k);
  std::vector<double> lat(nk), lon(nk), vm(nk), rm(static_cast<std::size_t>(n)),
      b(static_cast<std::size_t>(n)), lp(np), op(np);
  for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
    double a = 24.0, o = -88.0;
    for (std::size_t j = 0; j < static_cast<std::size_t>(k); ++j) {
      a += dlat(rng);
      o += dlon(rng);
      lat[i * 7 + j] = a;
      lon[i * 7 + j] = o;
      vm[i * 7 + j] = uv(rng);
    }
    rm[i] = urm(rng);
    b[i] = ub(rng);
  }
  for (std::size_t p = 0; p < static_cast<std::size_t>(np); ++p) {
    lp[p] = ula(rng);
    op[p] = ulo(rng);
  }
  // Land north of 30N (a stand-in for the coast; decay is a negligible part of the cost).
  std::vector<std::uint8_t> land(100 * 100, 0);
  for (std::size_t i = 80; i < 100; ++i)
    for (std::size_t j = 0; j < 100; ++j) land[i * 100 + j] = 1;
  const landfall::LandGridView g{10.0, -100.0, 0.25, 0.4, 100, 100, land.data()};
  const landfall::StormArrays s{node.data(), k, lat.data(), lon.data(), vm.data(),
                                rm.data(),   b.data(), n};
  std::vector<double> out(static_cast<std::size_t>(n * np));
  const bool ok = reference
                      ? landfall::max_wind_reference(s, lp.data(), op.data(), np, g, 0.5, true,
                                                     out.data())
                      : landfall::max_wind(s, lp.data(), op.data(), np, g, 0.5, true, out.data());
  double sum = 0.0;
  for (const double v : out) sum += v;
  std::printf("%s ok=%d evaluations=%lld checksum=%.10g\n", argv[2], ok ? 1 : 0,
              static_cast<long long>(n * 241 * np), sum);
  return ok ? 0 : 1;
}
