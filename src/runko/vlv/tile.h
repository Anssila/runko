#pragma once

#include "vlasov_grid.h"
#include "runko/emf/tile.h"
#include "runko/tools/config_parser.h"


namespace vlv{

template<std::size_t D>
class Tile : virtual public emf::Tile<D> {

  using value_type = VlasovGrid::value_type;
  using VDF = VlasovGrid::VelocityDistributionFunction;

  using SpatialGrid = tyvi::mdgrid_buffer<
    std::vector<DenseGrid>, 
    std::extents<std::size_t>, 
    std::layout_right, 
    std::extents<std::size_t,std::dynamic_extent,std::dynamic_extent,std::dynamic_extent>, 
    std::layout_right>;

public:
  explicit Tile(
    std::array<std::size_t, 3> tile_grid_indices,
    const toolbox::ConfigParser& config);

private:
    SpatialGrid grid_;

public:
    VlasovGrid& GetVelGrid(runko::index_t x, runko::index_t y, runko::index_t z);
    void SetVelGrid(runko::index_t x, runko::index_t y, runko::index_t z, VDF distribution);
    void DebugAccelerate(runko::index_t x, runko::index_t y, runko::index_t z, double ax, double ay, double az, double dt);

};

} // namespace vlv