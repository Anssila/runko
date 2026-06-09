#pragma once

#include "vlasov_grid.h"
#include "runko/emf/tile.h"
#include "runko/tools/config_parser.h"


namespace vlv{

template<std::size_t D>
class Tile : virtual public emf::Tile<D> {

  using value_type = VlasovGrid::value_type;
  using VDF = VlasovGrid::VelocityDistributionFunction;
public:
  explicit Tile(
    std::array<std::size_t, 3> tile_grid_indices,
    const toolbox::ConfigParser& config);

private:
    DenseGrid vel_grid_; // TODO actually store a 3d grid of cells each of which having a velocity grid

public:
    auto& GetVelGrid() { return vel_grid_; }
    void SetVelGrid(VDF distribution);
    void DebugAccelerate(double ax, double ay, double az, double dt);

};

} // namespace vlv