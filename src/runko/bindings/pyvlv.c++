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

auto to_6darray(vlv::Tile<3,vlv::DenseGrid>::VlasovSnapshot snapshot){
  const auto mds = snapshot.mds();
  const auto grid_shape = std::array{mds.extent(0), mds.extent(1), mds.extent(2), mds.extent(3), mds.extent(4), mds.extent(5)};
  auto pygrid = py::array_t<double, py::array::c_style>(grid_shape);

  auto pygridv = pygrid.template mutable_unchecked<6>();

  for (const auto idx : tyvi::sstd::index_space(mds)){
    const auto [i,j,k,l,m,n] = idx;
    pygridv(i,j,k,l,m,n) = static_cast<double>(mds[idx][]);
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
    .def("SetVelDistribution", [] (vlv::Tile<3, vlv::DenseGrid>& tile, int x, int y, int z, vlv::Tile<3, vlv::DenseGrid>::VDF f, int species) {
      tile.SetVelGrid( 
        static_cast<runko::index_t>(x + emf::halo_size), 
        static_cast<runko::index_t>(y + emf::halo_size), 
        static_cast<runko::index_t>(z + emf::halo_size), 
        f,
        static_cast<runko::index_t>(species)
      );
    })
    .def("GetVelDistribution", [] (vlv::Tile<3, vlv::DenseGrid>& tile, int x, int y, int z, int species) {
        return to_ndarray(dynamic_cast<vlv::DenseGrid&>(tile.GetVelGrid(
          static_cast<runko::index_t>(x + emf::halo_size),
          static_cast<runko::index_t>(y + emf::halo_size),
          static_cast<runko::index_t>(z + emf::halo_size),
          static_cast<runko::index_t>(species)
        )));
    })
    .def("DebugAccelerate", [] (vlv::Tile<3, vlv::DenseGrid>& tile, int x, int y, int z, double ax, double ay, double az) {
      tile.DebugAccelerate( 
        static_cast<runko::index_t>(x + emf::halo_size),
        static_cast<runko::index_t>(y + emf::halo_size),
        static_cast<runko::index_t>(z + emf::halo_size),
        ax, ay, az
      );
    })
    .def("set_vlv", &vlv::Tile<3, vlv::DenseGrid>::set_vlv)
    .def("Translate", &vlv::Tile<3, vlv::DenseGrid>::Translate)
    .def("accelerate", &vlv::Tile<3, vlv::DenseGrid>::accelerate)
    .def("CleanUp", &vlv::Tile<3, vlv::DenseGrid>::CleanUp)
    .def("DebugBC", &vlv::Tile<3, vlv::DenseGrid>::DebugBC)
    .def("deposit_current", &vlv::Tile<3, vlv::DenseGrid>::deposit_current)
    .def("CalculateMoment", [] (vlv::Tile<3, vlv::DenseGrid>& tile, int x, int y, int z, vlv::Tile<3, vlv::DenseGrid>::MCF f, int species) {
      return tile.CalculateMoment( 
        static_cast<runko::index_t>(x + emf::halo_size), 
        static_cast<runko::index_t>(y + emf::halo_size), 
        static_cast<runko::index_t>(z + emf::halo_size), 
        f,
        static_cast<runko::index_t>(species)
      );
    })
    .def("get_vlv_snapshot", [] (vlv::Tile<3, vlv::DenseGrid>& tile, int species) {
      return to_6darray(tile.get_vlasov_snapshot(species));
    })
    .def("get_tot_energy_E", &vlv::Tile<3, vlv::DenseGrid>::get_tot_energy_E)
    .def_static("canonical_type", []() { return py::type::of<vlv::Tile<3, DenseGrid>>(); });


//   m_3d.def("_write_average_kinetic_energy", &pic::write_average_kinetic_energy);

}
} // namespace vlv