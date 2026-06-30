#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#include "runko/vlv/tile.h"
#include "runko/bindings/runko_cpp_bindings.h"

namespace {

namespace py = pybind11;

/// MVP implementation.
auto
  to_ndarray(vlv::DenseGrid &grid)
{

  const auto grid_shape = std::array { grid.GetMDS().extent(0),
                                       grid.GetMDS().extent(1),
                                       grid.GetMDS().extent(2)};
  

  auto pygrid = py::array_t<double, py::array::c_style>(grid_shape);

  auto pygridv = pygrid.template mutable_unchecked<3>();
  
  for(const auto mds = grid.GetMDS(); const auto idx: tyvi::sstd::index_space(mds)) {
    const auto [i, j, k] = idx;
    const auto F         = mds[idx][];

    pygridv(i, j, k) = F;
  }

  return pygrid;
}
}  // namespace

namespace vlv{
namespace py = pybind11;

void bind_vlv(  py::module& m_sub){

  py::module m_3d = m_sub.def_submodule("threeD", "3D specializations");

    py::class_<vlv::Tile<3, vlv::DenseGrid>, emf::Tile<3>, corgi::Tile<3>, std::shared_ptr<vlv::Tile<3,vlv::DenseGrid>>>(
    m_3d,
    "Tile")
    .def(
      py::init([](const std::array<std::size_t, 3> tile_grid_indices, const py::handle& h) {
        return vlv::Tile<3, vlv::DenseGrid>(tile_grid_indices, toolbox::ConfigParser(h));
      }))
    .def("SetVelDistribution", [] (vlv::Tile<3, vlv::DenseGrid>& tile, int x, int y, int z, vlv::Tile<3, vlv::DenseGrid>::VDF f) {
      tile.SetVelGrid( 
        static_cast<runko::index_t>(x + emf::halo_size), 
        static_cast<runko::index_t>(y + emf::halo_size), 
        static_cast<runko::index_t>(z + emf::halo_size), 
        f 
      );
    })
    .def("GetVelDistribution", [] (vlv::Tile<3, vlv::DenseGrid>& tile, int x, int y, int z) {
        return to_ndarray(dynamic_cast<vlv::DenseGrid&>(tile.GetVelGrid(
          static_cast<runko::index_t>(x + emf::halo_size),
          static_cast<runko::index_t>(y + emf::halo_size),
          static_cast<runko::index_t>(z + emf::halo_size)
        )));
    })
    .def("DebugAccelerate", [] (vlv::Tile<3, vlv::DenseGrid>& tile, int x, int y, int z, double ax, double ay, double az, double dt) {
      tile.DebugAccelerate( 
        static_cast<runko::index_t>(x + emf::halo_size),
        static_cast<runko::index_t>(y + emf::halo_size),
        static_cast<runko::index_t>(z + emf::halo_size),
        ax, ay, az, dt
      );
    })
    .def("Translate", &vlv::Tile<3, vlv::DenseGrid>::Translate)
    .def("CleanUp", &vlv::Tile<3, vlv::DenseGrid>::CleanUp)
    .def("DebugBC", &vlv::Tile<3, vlv::DenseGrid>::DebugBC)
    .def("deposit_current", &vlv::Tile<3, vlv::DenseGrid>::deposit_current)
    .def("CalculateMoment", [] (vlv::Tile<3, vlv::DenseGrid>& tile, int x, int y, int z, vlv::Tile<3, vlv::DenseGrid>::MCF f) {
      return tile.CalculateMoment( 
        static_cast<runko::index_t>(x + emf::halo_size), 
        static_cast<runko::index_t>(y + emf::halo_size), 
        static_cast<runko::index_t>(z + emf::halo_size), 
        f 
      );
    });


//   m_3d.def("_write_average_kinetic_energy", &pic::write_average_kinetic_energy);

}
} // namespace vlv