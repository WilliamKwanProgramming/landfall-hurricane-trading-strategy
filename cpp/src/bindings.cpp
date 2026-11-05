// pybind11 module landfall_cpp: the footprint Monte Carlo kernel for Python.
// Python wrapper and argument documentation: landfall.footprint.max_wind_cpp.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "landfall/footprint.hpp"

namespace py = pybind11;
using namespace py::literals;

namespace {

using Dbl = py::array_t<double, py::array::c_style | py::array::forcecast>;
using U8 = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

void need(bool ok, const std::string& what) {
  if (!ok) throw std::invalid_argument(what);
}

py::array_t<double> max_wind_py(const Dbl& node_h, const Dbl& lat, const Dbl& lon,
                                const Dbl& vmax, const Dbl& rm, const Dbl& b, const Dbl& lat_p,
                                const Dbl& lon_p, double lat0, double lon0, double dlat,
                                double dlon, const U8& land, double dt_h, bool decay) {
  need(node_h.ndim() == 1, "node_h must be 1-D");
  need(lat.ndim() == 2 && lon.ndim() == 2 && vmax.ndim() == 2, "lat, lon, vmax must be 2-D");
  const auto n = lat.shape(0);
  const auto k = lat.shape(1);
  need(k == node_h.shape(0), "lat must have one column per node");
  need(lon.shape(0) == n && lon.shape(1) == k && vmax.shape(0) == n && vmax.shape(1) == k,
       "lat, lon, vmax must have the same shape");
  need(rm.ndim() == 1 && rm.shape(0) == n && b.ndim() == 1 && b.shape(0) == n,
       "rm and b must be 1-D with one value per storm");
  need(lat_p.ndim() == 1 && lon_p.ndim() == 1 && lat_p.shape(0) == lon_p.shape(0),
       "lat_p and lon_p must be 1-D of equal length");
  need(land.ndim() == 2, "land must be 2-D");
  const auto p = lat_p.shape(0);
  landfall::StormArrays s{node_h.data(), k, lat.data(), lon.data(), vmax.data(), rm.data(),
                          b.data(), n};
  landfall::LandGridView g{lat0, lon0, dlat, dlon, land.shape(0), land.shape(1), land.data()};
  py::array_t<double> out({n, p});
  bool ok = false;
  {
    py::gil_scoped_release release;
    ok = landfall::max_wind(s, lat_p.data(), lon_p.data(), p, g, dt_h, decay,
                            out.mutable_data());
  }
  need(ok, "node_h must have at least two strictly increasing values and dt_h must be > 0");
  return out;
}

py::array_t<double> exceedance_py(const Dbl& maxwind, const Dbl& thresholds) {
  need(maxwind.ndim() == 2 && thresholds.ndim() == 1, "maxwind 2-D, thresholds 1-D");
  const auto n = maxwind.shape(0);
  const auto p = maxwind.shape(1);
  const auto t = thresholds.shape(0);
  py::array_t<double> out({t, p});
  landfall::exceedance(maxwind.data(), n, p, thresholds.data(), t, out.mutable_data());
  return out;
}

}  // namespace

PYBIND11_MODULE(landfall_cpp, m) {
  m.doc() = "C++ footprint Monte Carlo kernel (twin of landfall.footprint.max_wind_numpy)";
  m.def("max_wind", &max_wind_py, "node_h"_a, "lat"_a, "lon"_a, "vmax"_a, "rm"_a, "b"_a,
        "lat_p"_a, "lon_p"_a, "lat0"_a, "lon0"_a, "dlat"_a, "dlon"_a, "land"_a, "dt_h"_a,
        "decay"_a);
  m.def("exceedance", &exceedance_py, "maxwind"_a, "thresholds"_a);
  m.def("symmetric_profile", &landfall::symmetric_profile, "r_km"_a, "vm_kt"_a, "rm_km"_a,
        "b"_a);
  m.def("inland_decay", &landfall::inland_decay, "v0_kt"_a, "hours_since_landfall"_a);
  m.def("wind_speed", &landfall::wind_speed, "lat_c"_a, "lon_c"_a, "vmax_kt"_a, "rm_km"_a,
        "b"_a, "ce_kmh"_a, "cn_kmh"_a, "lat_p"_a, "lon_p"_a);
}
