#include "vlasov_grid.h"
#include "runko/tools/vector.h"
#include "thrust/execution_policy.h"
#include "thrust/host_vector.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/reduce.h"
namespace vlv {

void
  DenseGrid::set_size(runko::index_t Nx, runko::index_t Ny, runko::index_t Nz)
{
  extents_  = { Nx, Ny, Nz };
  grid_     = std::make_unique<VelGrid>(Nx, Ny, Nz);
  new_grid_ = std::make_unique<VelGrid>(Nx, Ny, Nz);
};

std::array<runko::index_t, 3>
  DenseGrid::get_inds_from_vel(std::array<value_type, 3> u) const
{
  auto single_ax =
    [](value_type u, runko::index_t ex, value_type deltaU) -> runko::index_t {
    return static_cast<runko::index_t>(
      u / deltaU + static_cast<value_type>(ex - 1) * static_cast<value_type>(0.5f));
  };

  return std::array<runko::index_t, 3> { single_ax(u[0], extents_[0], u_res_[0]),
                                         single_ax(u[1], extents_[1], u_res_[1]),
                                         single_ax(u[2], extents_[2], u_res_[2]) };
}

std::array<vlv::VlasovGrid::value_type, 3>
  DenseGrid::get_vel_from_inds(std::array<runko::index_t, 3> inds) const
{
  auto single_ax =
    [](runko::index_t ind, runko::index_t ex, value_type deltaU) -> value_type {
    return (static_cast<value_type>(ind) - static_cast<value_type>(ex - 1) / 2) *
           deltaU;
  };

  return std::array<value_type, 3> { single_ax(inds[0], extents_[0], u_res_[0]),
                                     single_ax(inds[1], extents_[1], u_res_[1]),
                                     single_ax(inds[2], extents_[2], u_res_[2]) };
}

void
  DenseGrid::set_u_res(std::array<value_type, 3> res)
{
  u_res_ = res;
  for(size_t i = 0; i < 3ul; i++) {
    u_max_[i] = static_cast<value_type>(extents_[i] - 1ul) *
                static_cast<value_type>(0.5f) * u_res_[i];
  }
}

void
  DenseGrid::set_u_max(std::array<value_type, 3> max)
{
  u_max_ = max;
  for(size_t i = 0; i < 3ul; i++) {
    u_res_[i] = u_max_[i] / static_cast<value_type>(extents_[i] - 1ul) *
                static_cast<value_type>(2.0f);
  }
}

void
  DenseGrid::accelerate(const tyvi::mdgrid_work& w, value_type dv, runko::index_t ax)
{
  value_type shift_ind = dv / u_res_[ax];  // how many indicies we shift by

  // the relative index of the cell with lower index we need to update
  value_type min = sstd::floor(shift_ind);

  // the other relative index we need to update for every cell
  value_type max = sstd::ceil(shift_ind);

  // interpolation values for the two cells the fluid ends up in
  value_type t_min = min - shift_ind;
  value_type t_max = max - shift_ind;

  // a vector of the direction to accelerate in (defined by ax)
  toolbox::Vec3 dir = toolbox::Vec3(int32_t { 0 }, int32_t { 0 }, int32_t { 0 });
  dir[ax]           = int32_t { 1 };

  int32_t min_ind = static_cast<int32_t>(min);
  int32_t max_ind = static_cast<int32_t>(max);

  auto min_rel = dir * min_ind;  // relative position of min
  auto max_rel = dir * max_ind;  // relative position of max

  std::array<runko::index_t, 3> extents = extents_;

  auto kernel =
    [=, grid_mds = grid_->mds(), new_grid_mds = new_grid_->mds()](const auto& idx) {
      toolbox::Vec3 ind_vec = toolbox::Vec3(
        static_cast<int32_t>(idx[0]),
        static_cast<int32_t>(idx[1]),
        static_cast<int32_t>(idx[2]));

      auto interpolation_values = std::vector<value_type>();
      interpolation_values.push_back(grid_mds[idx][]);

      auto min_vec = ind_vec + min_rel;
      auto max_vec = ind_vec + max_rel;

      auto min_inds = std::array<int32_t, 3> { min_vec[0], min_vec[1], min_vec[2] };
      auto max_inds = std::array<int32_t, 3> { max_vec[0], max_vec[1], max_vec[2] };

      // Clamp the indices to make sure fluid doesn't disappear and we don't access
      // outside the buffer
      for(runko::index_t ax = 0; ax < 3ul; ax++) {
        min_inds[ax] = sstd::clamp(
          static_cast<int32_t>(min_inds[ax]),
          int32_t { 0 },
          static_cast<int32_t>(extents[ax]) - int32_t { 1 });
        max_inds[ax] = sstd::clamp(
          static_cast<int32_t>(max_inds[ax]),
          int32_t { 0 },
          static_cast<int32_t>(extents[ax]) - int32_t { 1 });
      }

      new_grid_mds[min_inds][] += interpolate(interpolation_values, t_min, 0);
      // Only add the second contribution if they are to different cells
      if(min_ind != max_ind)
        new_grid_mds[max_inds][] += interpolate(interpolation_values, t_max, 0);
    };

  w.for_each_index(*new_grid_, std::move(kernel));
}

void
  DenseGrid::translate(
    const tyvi::mdgrid_work& w,
    std::vector<VlasovGrid*> neighbors,
    value_type cfl,
    runko::index_t ax)
{
  auto mds_grids     = std::vector<decltype(grid_->mds())>();
  auto mds_new_grids = std::vector<decltype(new_grid_->mds())>();
  for(auto neighbor: neighbors) {
    mds_new_grids.push_back(static_cast<DenseGrid*>(neighbor)->new_grid_->mds());
    mds_grids.push_back(static_cast<DenseGrid*>(neighbor)->grid_->mds());
  }

  auto kernel = [this, mds_grids, mds_new_grids, cfl, ax](const auto& idx) {
    const auto u = toolbox::Vec3(this->get_vel_from_inds(idx));
    const value_type invGamma =
      value_type { 1 } / sstd::sqrt(value_type { 1 } + toolbox::dot(u, u));

    // how much we need to move in units of cells
    const value_type delta = u[ax] * invGamma * cfl;

    // the relative index of the cell with lower index we need to update
    value_type min = sstd::floor(delta);
    value_type max = sstd::ceil(delta);  // the other relative index we need to update

    // interpolation values for the two cells the fluid ends up in
    value_type t_min = min - delta;
    value_type t_max = max - delta;

    // calculate the indices of the relevant mds in terms of the given vector of grids
    int32_t min_ind =
      static_cast<int32_t>(min) + static_cast<int32_t>((mds_grids.size() - 1) / 2);
    int32_t max_ind =
      static_cast<int32_t>(max) + static_cast<int32_t>((mds_grids.size() - 1) / 2);

    // TODO: add support for higher order interpolations, i.e. more values here
    auto interpolation_values = std::vector<value_type>();
    interpolation_values.push_back(mds_grids[(mds_grids.size() - 1) / 2][idx][]);

    mds_new_grids[min_ind][idx][] += interpolate(interpolation_values, t_min, 0);
    // Only add the second contribution if they are to different cells
    if(min_ind != max_ind)
      mds_new_grids[max_ind][idx][] += interpolate(interpolation_values, t_max, 0);
  };

  w.for_each_index(*grid_, std::move(kernel));
}

void
  DenseGrid::recv_data(const tyvi::mdgrid_work& w, const VlasovGrid& orig)
{
  try {
    const DenseGrid& origin = dynamic_cast<const DenseGrid&>(orig);

    auto kernel = [dest_mds   = grid_->mds(),
                   source_mds = origin.grid_->mds()](const auto& idx) {
      // Simply copy the values
      dest_mds[idx][] = source_mds[idx][];
    };

    w.for_each_index(*grid_, std::move(kernel));

  } catch(const std::bad_cast& e) {
    throw std::runtime_error(
      "Cannot recv data from a VlasovGrid of a different type!\n");
  }
}

void
  DenseGrid::clean_up(const tyvi::mdgrid_work& w)
{
  // Set grid to zeros in order to have a clean new_grid after the swap
  w.for_each_index(*grid_, [grid_mds = grid_->mds()](const auto& idx) {
    grid_mds[idx][] = static_cast<value_type>(0.0f);
  });

  std::swap(grid_, new_grid_);
}

vlv::VlasovGrid::value_type
  DenseGrid::debug_get_fluid(std::array<runko::index_t, 3> inds) const
{
  const auto w = tyvi::mdgrid_work {};
  w.sync_to_staging(*grid_).sync_to_staging(*new_grid_).wait();
  return grid_->staging_mds()[inds][];
}


vlv::VlasovGrid::value_type
  DenseGrid::calculate_moment(
    const tyvi::mdgrid_work& w,
    MomentCalculationFunction func)
{
  namespace rn           = std::ranges;
  const auto grid_mds    = grid_->mds();
  const auto index_space = tyvi::sstd::index_space(grid_mds);

  // lambda to calculate a single part of the integral
  const auto calculate_integral = [this, grid_mds, func](const auto idx) {
    const auto f = grid_mds[idx][];
    const auto u = toolbox::Vec3(this->get_vel_from_inds(idx));

    const value_type gamma = sstd::sqrt(value_type { 1 } + toolbox::dot(u, u));

    return f * static_cast<value_type>(func(u[0], u[1], u[2], gamma));
  };

  // create a transform iterator to apply the lambda to every point in the velocity space
  const auto v_iterator_begin =
    thrust::make_transform_iterator(index_space.begin(), calculate_integral);

  const auto v_iterator_end = rn::next(v_iterator_begin, rn::size(index_space));

  // multiply by resolutions to normalize the integral (d^3u = du_x*du_y*du_z)
  const auto du3 = u_res_[0] * u_res_[1] * u_res_[2];

  // finally reduce the integral using the iterator
  return thrust::reduce(w.on_this(), v_iterator_begin, v_iterator_end) * du3;
}

constexpr vlv::VlasovGrid::value_type
  VlasovGrid::interpolate(
    std::vector<value_type>& values,
    value_type t,
    runko::index_t order)
{

  // The interpolator takes in values describing the distribution around a center point
  // (the middle value of "values" that should always contain exactly 2*order+1 values)
  // and interpolates the approximate distribution (to order "order") at t that is
  // measured relative to the center of the middle cell.

  switch(order) {
    case 0:
      // 0th order interpolation; distribution is modeled as a "staircase" and a simple
      // linear interpolation will be performed
      return values[0] * (static_cast<value_type>(1.0f) - sstd::abs(t));
    default:
      std::stringstream msg;
      msg << "Interpolator of order " << order << " is not implemented!\n";
      throw std::runtime_error(msg.str());
  }
}


void
  DenseGrid::init_zero()
{
  const auto staging_mds = grid_->staging_mds();
  for(const auto idx: tyvi::sstd::index_space(staging_mds)) {
    staging_mds[idx][] = static_cast<value_type>(0.0f);
  }
  const auto new_staging_mds = new_grid_->staging_mds();
  for(const auto idx: tyvi::sstd::index_space(new_staging_mds)) {
    new_staging_mds[idx][] = static_cast<value_type>(0.0f);
  }

  const auto w = tyvi::mdgrid_work {};
  w.sync_from_staging(*grid_).sync_from_staging(*new_grid_).wait();
}

void
  DenseGrid::init_delta(std::array<value_type, 3> v)
{
  const auto staging_mds = grid_->staging_mds();

  std::array<runko::index_t, 3> v_inds = get_inds_from_vel(v);

  for(const auto idx: tyvi::sstd::index_space(staging_mds)) {
    // If the index is the one corresponding to the given velocity, we set density to
    // one, otherwise to zero
    staging_mds[idx][] = static_cast<value_type>(idx == v_inds ? 1.0f : 0.0f);
  }

  // set the new_grid to zeros to have a clean buffer
  const auto new_staging_mds = new_grid_->staging_mds();
  for(const auto idx: tyvi::sstd::index_space(new_staging_mds)) {
    new_staging_mds[idx][] = static_cast<value_type>(0.0f);
  }

  const auto w = tyvi::mdgrid_work {};
  w.sync_from_staging(*grid_).sync_from_staging(*new_grid_).wait();
}


void
  DenseGrid::set_grid_data(VlasovGrid::VelocityDistributionFunction distribution)
{
  const auto staging_mds = grid_->staging_mds();

  for(const auto idx: tyvi::sstd::index_space(staging_mds)) {
    // Get the associated velocity
    auto u = get_vel_from_inds(idx);

    // convert to doubles for the distribution function and then back to value_type
    staging_mds[idx][] = static_cast<value_type>(distribution(
      static_cast<double>(u[0]),
      static_cast<double>(u[1]),
      static_cast<double>(u[2])));
  }

  const auto w = tyvi::mdgrid_work {};
  w.sync_from_staging(*grid_).wait();
}


vlv::VlasovGrid::value_type
  DenseGrid::debug_get_total_fluid() const
{
  const auto w = tyvi::mdgrid_work {};
  w.sync_to_staging(*grid_).sync_to_staging(*new_grid_).wait();
  const auto staging_mds = grid_->staging_mds();
  value_type tot         = static_cast<value_type>(0.0f);
  for(const auto idx: tyvi::sstd::index_space(staging_mds)) {
    tot += staging_mds[idx][];
  }
  return tot;
}

}  // namespace vlv
