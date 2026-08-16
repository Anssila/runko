#pragma once

#include "vlasov_grid.h"

namespace vlv {

template<VelGridType VGrid>
struct VlasovContainerArgs {
  double charge;
  double mass;
  std::array<runko::index_t, 3> spatial_extents;
  std::array<runko::index_t, 3> velocity_extents;

  // The values used to initialize the velocity grid (passed into init_func for every
  // VlasovGrid)
  std::array<vlv::VlasovGrid::value_type, 3> u_init;

  // The function used to initialize the velocity grid
  std::function<void(VGrid&, std::array<vlv::VlasovGrid::value_type, 3>)> init_func;
};

// Container for holding the Vlasov fluid of a single particle species
// Contains a (spatial) grid of VlasovGrid objects and some common properties of the
// particles/fluid
template<VelGridType VGrid>
class VlasovContainer {
public:
  using value_type = vlv::VlasovGrid::value_type;

  // Type for storing the velocity grids of each cell of the tile in an mdgrid_buffer
  using SpatialGrid = tyvi::mdgrid_buffer<
    std::vector<VGrid>,
    std::extents<std::size_t>,
    std::layout_right,
    std::extents<
      std::size_t,
      std::dynamic_extent,
      std::dynamic_extent,
      std::dynamic_extent>,
    std::layout_right>;

private:
  const double charge_;
  const double mass_;

  // The actual grid of VlasovGrids
  SpatialGrid grid_;

public:
  double charge() const { return charge_; }
  double mass() const { return mass_; }

  explicit VlasovContainer(VlasovContainerArgs<VGrid> args);

  auto mds() { return grid_.mds(); }
  auto mds() const { return grid_.mds(); }
  auto extents() const { return grid_.grid_extents(); }
};

}  // namespace vlv