#pragma once

#include "runko/communication_common.h"
#include "runko/emf/tile.h"
#include "runko/tools/config_parser.h"
#include "vlasov_container.h"
#include "vlasov_grid.h"


namespace vlv {

// Templated Tile class that has the VlasovGrid implementation (DenseGrid, ...)
// templated as well as the dimension
template<std::size_t D, VelGridType VGrid>
class Tile : virtual public emf::Tile<D> {
public:
  using value_type = VlasovGrid::value_type;

  // Type for functions that define velocity space distributions for initialization
  using VDF = VlasovGrid::VelocityDistributionFunction;

  // Type for functions that calculate moments of the velocity space
  using MCF = VlasovGrid::MomentCalculationFunction;

  // Function for initializing the whole 6D vlasov fluid
  using VlasovInitFunc =
    std::function<double(double, double, double, double, double, double)>;

  // Type for storing a full snapshot of the 6D Vlasov fluid
  using VlasovSnapshot = tyvi::mdgrid_buffer<
    std::vector<value_type>,
    std::extents<std::size_t>,
    std::layout_right,
    std::extents<
      std::size_t,
      std::dynamic_extent,
      std::dynamic_extent,
      std::dynamic_extent,
      std::dynamic_extent,
      std::dynamic_extent,
      std::dynamic_extent>,
    std::layout_right>;

public:
  explicit Tile(
    std::array<std::size_t, 3> tile_grid_indices,
    const toolbox::ConfigParser& config);

protected:
  // Vector of all the particle species containers
  // The containers contain the spatial grids of the VlasovGrids as well as some
  // information about the species in question (mass, charge, ...)
  std::vector<VlasovContainer<VGrid>> containers_;

  // Get the halo size defined for emf Tiles
  static constexpr runko::index_t halo_size =
    static_cast<runko::index_t>(emf::halo_size);

  // The spatial extents of the tile, including the halo regions
  const std::array<runko::index_t, 3> extents_;

  // The velocity extents of each VlasovGrid, there are no halo regions for velocity
  // space
  const std::array<runko::index_t, 3> velocity_extents_;

  // The spatial offset of this tile in units of spatial cells
  const std::array<runko::index_t, 3> spatial_offset_;

  // TODO: do io properly using MPI-IO
  // index of the next snapshot
  std::size_t vlv_snapshot_index;

  std::string io_outdir;

  // function for getting the submdspan not containing the halo regions
  template<typename MDS>
  auto nonhalo_submds(MDS&& mds) const
  {
    const auto extents = this->yee_lattice_.extents_wout_halo();
    const auto x       = std::tuple { halo_size, halo_size + extents[0] };
    const auto y       = std::tuple { halo_size, halo_size + extents[1] };
    const auto z       = std::tuple { halo_size, halo_size + extents[2] };

    return std::submdspan(std::forward<MDS>(mds), x, y, z);
  }

  // Check that given indices are inside the full extents of the tile
  void assert_inside(std::array<runko::index_t, 3> idx) const;

public:
  // Get a reference to the velocity distribution (VlasovGrid) of a specific cell
  VlasovGrid& get_vel_grid(
    runko::index_t x,
    runko::index_t y,
    runko::index_t z,
    runko::index_t species);

  // Set the velocity distribution of a specific cell
  void set_vel_grid(
    runko::index_t x,
    runko::index_t y,
    runko::index_t z,
    VDF distribution,
    runko::index_t species);

  // Accelerate the plasma of a cell homogeneously by a non-physical acceleration for
  // debug purposes
  void debug_accelerate(
    runko::index_t x,
    runko::index_t y,
    runko::index_t z,
    double ax,
    double ay,
    double az);

  // Apply translation in regular space to all the cell in the tile
  void translate();

  void clean_up();  // Clean and swap buffers to be ready for the next iteration

  // Apply periodic boundary conditions for this tile, emulates (local) communication
  // between tiles
  void debug_BC();

  // Calculate a moment (specified by func) of the velocity space of the cell at x,y,z
  value_type calculate_moment(
    runko::index_t x,
    runko::index_t y,
    runko::index_t z,
    MCF func,
    runko::index_t species);

  // Calculate and deposit the current into the yee lattice
  void deposit_current();

  // Accelerate the fluid using the electric field
  void accelerate();

  // Initialize the Vlasov fluid using a full 6D function for the species given
  void set_vlv(VlasovInitFunc func, runko::index_t species);

  // Get a snapshot of the full 6D phase space of this Tile
  VlasovSnapshot get_vlasov_snapshot(runko::index_t species);

  // Write a snapshot of the full 6D phase space to disk
  // TODO: make a proper MPI-IO writer to do this
  void write_vlv_snapshot();

  // Get the total energy in the E-field, just uses emf::Tile's func
  double get_tot_energy_E() { return this->total_energy_E(); }

  void local_communication(
    const corgi::Tile<D>& /* other */,
    const std::array<int, D> dir_to_other,
    const int /* mode */
    ) override;

  std::vector<mpi4cpp::mpi::request>
    send_data(mpi4cpp::mpi::communicator& /*comm*/, int dest, int mode, int tag)
      override;

  std::vector<mpi4cpp::mpi::request>
    recv_data(mpi4cpp::mpi::communicator& /*comm*/, int orig, int mode, int tag)
      override;
};

}  // namespace vlv