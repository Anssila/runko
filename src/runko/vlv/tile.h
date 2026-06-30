#pragma once

#include "vlasov_grid.h"
#include "runko/emf/tile.h"
#include "runko/tools/config_parser.h"


namespace vlv{

// Concept for templating Tiles based on the implementation of VlasovGrid that it uses
template <typename VGrid>
concept VelGridType = std::derived_from<VGrid, VlasovGrid>;

// Templated Tile class that has the VlasovGrid implementation (DenseGrid, ...) templated as well as the dimension
template<std::size_t D, VelGridType VGrid>
class Tile : virtual public emf::Tile<D> {
public:
  using value_type = VlasovGrid::value_type;
  using VDF = VlasovGrid::VelocityDistributionFunction; // Type for functions that define velocity space distributions for initialization
  using MCF = VlasovGrid::MomentCalculationFunction; // Type for functions that calculate moments of the velocity space

  // Type for storing the velocity grids of each cell of the tile in an mdgrid_buffer
  using SpatialGrid = tyvi::mdgrid_buffer<
    std::vector<VGrid>, 
    std::extents<std::size_t>, 
    std::layout_right, 
    std::extents<std::size_t,std::dynamic_extent,std::dynamic_extent,std::dynamic_extent>, 
    std::layout_right>;

public:
  explicit Tile(
    std::array<std::size_t, 3> tile_grid_indices,
    const toolbox::ConfigParser& config);

protected:
  SpatialGrid grid_; // the grid of VlasovGrids for each cell

  static constexpr runko::index_t halo_size = static_cast<runko::index_t>(emf::halo_size);
  const std::array<runko::index_t, 3> extents_;

  // function for getting the sub mdspan not containing the halo regions
  template<typename MDS>
  auto nonhalo_submds(MDS&& mds) const
  {
    const auto extents = this->yee_lattice_.extents_wout_halo();
    const auto x = std::tuple { halo_size, halo_size + extents[0] };
    const auto y = std::tuple { halo_size, halo_size + extents[1] };
    const auto z = std::tuple { halo_size, halo_size + extents[2] };

    return std::submdspan(std::forward<MDS>(mds), x, y, z);
  }

  void AssertInside(std::array<runko::index_t,3> idx) const;

public:
    VlasovGrid& GetVelGrid(runko::index_t x, runko::index_t y, runko::index_t z); // Get a reference to the velocity distribution (VlasovGrid) of a specific cell
    void SetVelGrid(runko::index_t x, runko::index_t y, runko::index_t z, VDF distribution); // Set the velocity distribution of a specific cell
    void DebugAccelerate(runko::index_t x, runko::index_t y, runko::index_t z, double ax, double ay, double az, double dt); // Accelerate the plasma of a cell homogeneously by a non-physical acceleration for debug purposes
    void Translate(); // Apply translation in regular space to all the cell in the tile (Only in the z-direction for now!)
    void CleanUp(); // Clean and swap buffers to be ready for the next iteration
    void DebugBC(); // Apply periodic boundary conditions for this tile, emulates (local) communication between tiles
    value_type CalculateMoment(runko::index_t x, runko::index_t y, runko::index_t z, MCF func); // Calculate a moment (specified by func) of the velocity space of the cell at x,y,z
    void deposit_current(); // Calculate and deposit the current into the yee lattice
  };

} // namespace vlv