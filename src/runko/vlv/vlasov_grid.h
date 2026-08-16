#pragma once

#include "../mdgrid_common.h"

namespace vlv {

// Base class for different kinds of vlasov grid implementations
class VlasovGrid {
public:
  using value_type = float;

  // Function to set the VDF, inputs are velocity coordinates in x, y, and z directions
  // and output is the amount of phase fluid
  using VelocityDistributionFunction = std::function<double(double, double, double)>;

  // Function to calculate a moment of the velocity space, inputs are velocity
  // coordinates (u_x,u_y,u_z) as well as a pre-calculated gamma (sqrt(1+u**2))
  using MomentCalculationFunction =
    std::function<double(double, double, double, double)>;

protected:
  // returns interpolation result to given order (not properly implemented)
  static constexpr value_type interpolate(
    std::vector<value_type>& values,
    value_type t,
    const runko::index_t order = 0);

public:
  // Function to initialize the velocity distribution to zeros
  virtual void init_zero() = 0;

  // Function to initialize the velocity distribution to a delta function around the
  // specified velocity v (sets the cell in v-space containing v to 1.0 and rest to 0.0)
  virtual void init_delta(std::array<value_type, 3> v) = 0;

  virtual value_type debug_get_total_fluid() const = 0;

  // Set the velocity grid data using the given distribution function
  virtual void set_grid_data(VelocityDistributionFunction distribution) = 0;

  // function to set the size of the grid, actually creates the underlying buffers
  virtual void set_size(runko::index_t Nx, runko::index_t Ny, runko::index_t Nz) = 0;

  virtual void set_u_max(std::array<value_type, 3> max) = 0;
  virtual void set_u_res(std::array<value_type, 3> res) = 0;

  // Translate the fluid in the z-axis from this VlasovGrid to neighboring grids
  // depending on the velocity space coordinates
  virtual void translate(
    const tyvi::mdgrid_work& w,
    std::vector<VlasovGrid*> neighbors,
    value_type cfl,
    runko::index_t ax) = 0;

  // clean_up the old buffer and swap
  virtual void clean_up(const tyvi::mdgrid_work& w) = 0;

  // A non-async overload of clean_up
  virtual void clean_up() = 0;

  // Receive the data of a VlasovGrid and update this VlasovGrids data to match that
  virtual void recv_data(const tyvi::mdgrid_work& w, const VlasovGrid& orig) = 0;

  // Generalized function for calculating moments of the distribution velocity
  // distribution.
  virtual value_type
    calculate_moment(const tyvi::mdgrid_work& w, MomentCalculationFunction func) = 0;

  // accelerate the velocity space values in the ax-direction by dv
  virtual void
    accelerate(const tyvi::mdgrid_work& w, value_type dv, runko::index_t ax) = 0;
};


// Simple densely stored grid implementation of a vlasov grid
class DenseGrid : public VlasovGrid {
public:
  using VelGrid = runko::ScalarGrid<value_type>;

private:
  // The extents of the DenseGrid (there are no halo regions for the velocity space)
  std::array<runko::index_t, 3> extents_;

  // The max value for u that can be stored in the dense grid, for each axis
  std::array<value_type, 3> u_max_;

  // The resolution for u, i.e. what is the difference in u of neighboring
  // cells of the dense grid, for each axis
  std::array<value_type, 3> u_res_;

  // the actual grids that store the velocity space phase fluid
  // new_grid_ is used to update values and the pointers are swapped every time
  std::shared_ptr<VelGrid> grid_, new_grid_;

public:
  // default constructor to allow creating DenseGrids as elements of a buffer
  DenseGrid() = default;

  // overload to allow also specifying size while constructing
  DenseGrid(runko::index_t Nx, runko::index_t Ny, runko::index_t Nz)
  { set_size(Nx, Ny, Nz); }

  void set_size(runko::index_t Nx, runko::index_t Ny, runko::index_t Nz) override;

  void init_zero() override;
  void init_delta(std::array<value_type, 3>) override;

  // functions to set the maximum value of u stored in the DenseGrid or alternatively
  // the resolution (spacing) of values of u, for each axis separately
  void set_u_max(std::array<value_type, 3> max) override;
  void set_u_res(std::array<value_type, 3> res) override;

  void set_grid_data(VelocityDistributionFunction distribution) override;

  void translate(
    const tyvi::mdgrid_work& w,
    std::vector<VlasovGrid*> neighbors,
    value_type cfl,
    runko::index_t ax) override;

  void clean_up(const tyvi::mdgrid_work& w) override;
  void clean_up() override
  {
    const auto w = tyvi::mdgrid_work {};
    clean_up(w);
    w.wait();
  }

  void recv_data(const tyvi::mdgrid_work& w, const VlasovGrid& orig) override;

  value_type calculate_moment(
    const tyvi::mdgrid_work& w,
    MomentCalculationFunction func) override;

  // Helper function to get the indicies corresponding to a velocity in the sparse grid
  std::array<runko::index_t, 3> get_inds_from_vel(std::array<value_type, 3> u) const;

  // Helper function to get the velocity corresponding to a set of indicies in the dense
  // grid
  std::array<value_type, 3> get_vel_from_inds(std::array<runko::index_t, 3> inds) const;

  value_type debug_get_total_fluid() const override;

  // Debug function to get the fluid in a grid cell, this function is inefficient!
  value_type debug_get_fluid(std::array<runko::index_t, 3> inds) const;

  auto staging_mds() const
  {
    const auto w = tyvi::mdgrid_work {};
    w.sync_to_staging(*grid_).wait();
    return grid_->staging_mds();
  }

  auto mds() const { return grid_->mds(); }
  auto span() const { return grid_->span(); }
  auto get_extents() const { return extents_; }

  void
    accelerate(const tyvi::mdgrid_work& w, value_type dv, runko::index_t ax) override;
};

// Concept for templating Tiles (and VlasovContainers) based on the implementation of
// VlasovGrid that it uses
template<typename VGrid>
concept VelGridType = std::derived_from<VGrid, VlasovGrid>;

}  // namespace vlv