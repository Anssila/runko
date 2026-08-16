#include "tile.h"

#include <fstream>
namespace vlv {

template<std::size_t D, VelGridType VGrid>
Tile<D, VGrid>::Tile(
  std::array<std::size_t, 3> tile_grid_indices,
  const toolbox::ConfigParser& conf) :
  corgi::Tile<D>(),
  emf::Tile<D>(tile_grid_indices, conf),
  containers_(),
  extents_ {
    static_cast<runko::index_t>(
      conf.get_or_throw<std::vector<std::ptrdiff_t>>("n_cells_per_tile")[0]) + 2 * halo_size,
    static_cast<runko::index_t>(
      conf.get_or_throw<std::vector<std::ptrdiff_t>>("n_cells_per_tile")[1]) + 2 * halo_size,
    static_cast<runko::index_t>(
      conf.get_or_throw<std::vector<std::ptrdiff_t>>("n_cells_per_tile")[2]) + 2 * halo_size
  },
  velocity_extents_ {
    static_cast<runko::index_t>(
      conf.get_or_throw<std::vector<std::ptrdiff_t>>("v_grid_extents")[0]),
    static_cast<runko::index_t>(
      conf.get_or_throw<std::vector<std::ptrdiff_t>>("v_grid_extents")[1]),
    static_cast<runko::index_t>(
      conf.get_or_throw<std::vector<std::ptrdiff_t>>("v_grid_extents")[2])
  },
  spatial_offset_ {
    static_cast<runko::index_t>(
      conf.get_or_throw<std::vector<std::ptrdiff_t>>("n_cells_per_tile")[0] * tile_grid_indices[0]),
    static_cast<runko::index_t>(
      conf.get_or_throw<std::vector<std::ptrdiff_t>>("n_cells_per_tile")[1] * tile_grid_indices[1]),
    static_cast<runko::index_t>(
      conf.get_or_throw<std::vector<std::ptrdiff_t>>("n_cells_per_tile")[2] * tile_grid_indices[2])
  }
{

  using opt_args_t = std::optional<vlv::VlasovContainerArgs<VGrid>>;

  // Converter function to make sure the values given from Python are converted
  // correctly
  const auto converter = [](std::vector<double> v) {
    return std::array<vlv::VlasovGrid::value_type, 3> {
      static_cast<vlv::VlasovGrid::value_type>(v[0]),
      static_cast<vlv::VlasovGrid::value_type>(v[1]),
      static_cast<vlv::VlasovGrid::value_type>(v[2])
    };
  };

  // Return optional containing the arguments if q and m are found in conf.
  const auto make_opt_args =
    [&](const std::string& q_label, const std::string& m_label) -> opt_args_t 
  {
    const auto q_to_qm_tuple = [&](const auto q) {
      return conf.get<double>(m_label).transform(
        [&](const auto m) { return std::tuple { q, m }; });
    };

    const auto qm_tuple_to_args = [&](const auto qm) {
      const auto [q, m] = qm;
      return vlv::VlasovContainerArgs<VGrid> { .charge           = q,
                                                .mass             = m,
                                                .spatial_extents  = { 0, 0, 0 },
                                                .velocity_extents = { 0, 0, 0 },
                                                .u_init           = { 0.0, 0.0, 0.0 },
                                                .init_func = &VlasovGrid::set_u_max };
    };

    return conf.get<double>(q_label)
      .and_then(q_to_qm_tuple)
      .transform(qm_tuple_to_args);
  };

  // We get either u_max or u_res and deduce the other based on the extents
  // Based on which we get, we must pass the correct values and the correct function to the container
  auto u_max = conf.get<std::vector<double>>("u_max");
  auto u_res = conf.get<std::vector<double>>("u_res");

  // The function to call to init the VlasovGrid
  std::function<void(VGrid&, std::array<vlv::VlasovGrid::value_type, 3>)> init_func;

  // The values to pass to the init_func
  std::array<value_type, 3> init_values;

  // Check which initialization to use and initialize appropriately
  if(u_res.has_value() && !u_max.has_value()) {
    init_func   = &VGrid::set_u_res;
    init_values = converter(u_res.value());
  } else if(!u_res.has_value() && u_max.has_value()) {
    init_func   = &VGrid::set_u_max;
    init_values = converter(u_max.value());
  } else {
    std::stringstream msg;
    if(!u_res.has_value() && !u_max.has_value())
      msg << "Cannot create tile without either u_res or u_max!\n";
    else
      msg << "Cannot create tile with both u_res and u_max!\n";
    throw std::runtime_error(msg.str());
  }

  // Create containers for species until there are no more charges and masses given
  for(auto i = 0uz; true; ++i) {
    const auto q_label = std::format("q{}", i);
    const auto m_label = std::format("m{}", i);

    if(const auto vcontainer_args = make_opt_args(q_label, m_label)) {
      auto args             = vcontainer_args.value();
      args.spatial_extents  = extents_;
      args.velocity_extents = velocity_extents_;
      args.u_init           = init_values;
      args.init_func        = init_func;
      containers_.emplace_back(vlv::VlasovContainer<VGrid> { args });
    } else {
      break;
    }
  }

  // Init io index and outdir
  vlv_snapshot_index = 0;
  io_outdir = conf.get<std::string>("io_outdir").value_or(std::string("sim_out"));
}

template<std::size_t D, VelGridType VGrid>
void
  Tile<D, VGrid>::assert_inside(std::array<runko::index_t, 3> idx) const
{
  if(idx[0] >= extents_[0] || idx[1] >= extents_[1] || idx[2] >= extents_[2])
    throw std::runtime_error { std::format(
      "Indicies out of range! Got {}, {}, {} while extents are {}, {}, {}\n",
      idx[0],
      idx[1],
      idx[2],
      extents_[0],
      extents_[1],
      extents_[2]) };
}

template<std::size_t D, VelGridType VGrid>
void
  Tile<D, VGrid>::debug_accelerate(
    runko::index_t x,
    runko::index_t y,
    runko::index_t z,
    double ax,
    double ay,
    double az)
{
  auto ax_     = static_cast<value_type>(ax);
  auto ay_     = static_cast<value_type>(ay);
  auto az_     = static_cast<value_type>(az);
  const auto w = tyvi::mdgrid_work {};

  for(auto& species: containers_) {
    const auto mds = species.mds();
    const auto qpm = static_cast<value_type>(species.charge() / species.mass());
    auto idx       = std::array<runko::index_t, 3> { x, y, z };
    assert_inside(idx);
    mds[idx][].accelerate(w, qpm * ax_, 0);
    w.wait();
    mds[idx][].clean_up();
    mds[idx][].accelerate(w, qpm * ay_, 1);
    w.wait();
    mds[idx][].clean_up();
    mds[idx][].accelerate(w, qpm * az_, 2);
    w.wait();
    mds[idx][].clean_up();
  }
}

template<std::size_t D, VelGridType VGrid>
void
  Tile<D, VGrid>::set_vel_grid(
    runko::index_t x,
    runko::index_t y,
    runko::index_t z,
    VDF distribution,
    runko::index_t species)
{
  auto idx = std::array<runko::index_t, 3> { x, y, z };
  assert_inside(idx);
  const auto mds = containers_[species].mds();
  mds[idx][].set_grid_data(distribution);
}

template<std::size_t D, VelGridType VGrid>
VlasovGrid&
  Tile<D, VGrid>::get_vel_grid(
    runko::index_t x,
    runko::index_t y,
    runko::index_t z,
    runko::index_t species)
{
  auto idx = std::array<runko::index_t, 3> { x, y, z };
  assert_inside(idx);
  const auto mds = containers_[species].mds();
  return mds[idx][];
}

template<std::size_t D, VelGridType VGrid>
void
  Tile<D, VGrid>::set_vlv(Tile<D, VGrid>::VlasovInitFunc func, runko::index_t species)
{
  const auto nh_mds = nonhalo_submds(containers_[species].mds());
  for(auto idx: tyvi::sstd::index_space(nh_mds)) {
    const double x = static_cast<double>(idx[0] + spatial_offset_[0]) + 0.5;
    const double y = static_cast<double>(idx[1] + spatial_offset_[1]) + 0.5;
    const double z = static_cast<double>(idx[2] + spatial_offset_[2]) + 0.5;

    // Set the VlasovGrid data using a lambda that binds the spatial variables 
    // but not the velocity variables
    nh_mds[idx][].set_grid_data(
      [=](double ux, double uy, double uz) { return func(x, y, z, ux, uy, uz); });
  }
}


template<std::size_t D, VelGridType VGrid>
Tile<D, VGrid>::VlasovSnapshot
  Tile<D, VGrid>::get_vlasov_snapshot(runko::index_t species)
{
  // Get a full 6D snapshot of the phase space
  // Currently implemented fully on the host

  const auto nh_mds = nonhalo_submds(containers_[species].mds());
  auto snapshot     = VlasovSnapshot(
    nh_mds.extent(0),
    nh_mds.extent(1),
    nh_mds.extent(2),
    velocity_extents_[0],
    velocity_extents_[1],
    velocity_extents_[2]);

  const auto snapshot_mds = snapshot.mds();

  // TODO: properly handle other kinds of VlasovGrids
  static_assert(std::is_convertible_v<VGrid*, DenseGrid*>);

  for(auto idx: tyvi::sstd::index_space(nh_mds)) {
    const auto grid = static_cast<vlv::DenseGrid&>(get_vel_grid(
      static_cast<runko::index_t>(idx[0]) + emf::halo_size,
      static_cast<runko::index_t>(idx[1]) + emf::halo_size,
      static_cast<runko::index_t>(idx[2]) + emf::halo_size,
      species));

    const auto grid_mds = grid.staging_mds();

    for(auto jdx: tyvi::sstd::index_space(grid_mds)) {
      const auto snapshot_idx =
        std::array<std::size_t, 6> { idx[0], idx[1], idx[2], jdx[0], jdx[1], jdx[2] };

      snapshot_mds[snapshot_idx][] = grid_mds[jdx][];
    }
  }

  return snapshot;
}

template<std::size_t D, VelGridType VGrid>
void
  Tile<D, VGrid>::write_vlv_snapshot()
{
  auto filename = std::format(
    "{}/vlv_snapshot({},{},{})_{}.bin", 
    io_outdir, 
    this->index[0], 
    this->index[1], 
    this->index[2], 
    vlv_snapshot_index++);

  std::ofstream stream(filename, std::ios::binary);

  auto containers = containers_.size();

  // First write the amount of different species
  stream.write(reinterpret_cast<const char*>(&containers), sizeof(containers));

  // Write all the species
  for(runko::index_t i = 0; i < containers_.size(); i++) {
    // Get the snapshot of this species
    auto snapshot = get_vlasov_snapshot(i);
    auto mds      = snapshot.mds();
    auto size     = mds.size();

    // Get the full extents of the 6D phase space
    auto exs =
      std::array<std::size_t, 6> { mds.extent(0), mds.extent(1), mds.extent(2),
                                   mds.extent(3), mds.extent(4), mds.extent(5) };

    // Write the total number of phase space values corresponding to this species
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // Write the extents of the 6D space (we write only the non-halo region of spatial extents)
    stream.write(reinterpret_cast<const char*>(&exs[0]), sizeof(exs[0]) * 6);

    // Write the whole data of the snapshot
    auto span = snapshot.span();
    for (auto idx : tyvi::sstd::index_space(mds)){
      value_type val = mds[idx][];
      stream.write(reinterpret_cast<const char*>(&val), sizeof(value_type));
    }
  }
  stream.close();
}

template<std::size_t D, VelGridType VGrid>
void
  Tile<D, VGrid>::translate()
{

  const auto w = tyvi::mdgrid_work {};

  // Define the extents for each axis we need to update
  // We must update also the innermost cells of the halo region because we have a fwd
  // semi lagrangian scheme (only the innermost are enough because fluid mustn't move
  // more than a single cell in a single time step)
  const auto x = std::tuple { halo_size - 1, extents_[0] - halo_size + 1 };
  const auto y = std::tuple { halo_size - 1, extents_[1] - halo_size + 1 };
  const auto z = std::tuple { halo_size - 1, extents_[2] - halo_size + 1 };

  // Function to transform indices of the update mds to those of the full mds
  const auto getInds = [=](std::array<uint64_t, 3> idx) {
    return std::array<uint64_t, 3> { idx[0] + halo_size - 1,
                                     idx[1] + halo_size - 1,
                                     idx[2] + halo_size - 1 };
  };

  const auto translate_axis =
    [&, this](runko::index_t ax, value_type dt) {
      for(auto& species: this->containers_) {
        const auto mds = species.mds();

        const auto update_mds =
          std::submdspan(std::forward<decltype(mds)>(mds), x, y, z);

        for(auto idx: tyvi::sstd::index_space(update_mds)) {
          auto neighbors = std::vector<VlasovGrid*>();

          // TODO allow higher order reconstruction / interpolation by adding more neighbors
          auto neigh_idx = idx;
          neigh_idx[ax]--;
          neighbors.push_back(&mds[getInds(neigh_idx)][]);
          neigh_idx[ax]++;
          neighbors.push_back(&mds[getInds(neigh_idx)][]);
          neigh_idx[ax]++;
          neighbors.push_back(&mds[getInds(neigh_idx)][]);

          mds[getInds(idx)][]
            .translate(w, neighbors, static_cast<value_type>(this->cfl_) * dt, ax);
        }
      }
      w.wait();
      this->clean_up();
    };

  // TODO implement Strang-splitting
  translate_axis(0, 1.0f);
  translate_axis(1, 1.0f);
  translate_axis(2, 1.0f);
}

template<std::size_t D, VelGridType VGrid>
void
  Tile<D, VGrid>::clean_up()
{
  const auto w = tyvi::mdgrid_work {};
  for(auto& species: containers_) {
    const auto mds = species.mds();
    for(auto idx: tyvi::sstd::index_space(mds)) { mds[idx][].clean_up(w); }
  }
  w.wait();
}

template<std::size_t D, VelGridType VGrid>
Tile<D, VGrid>::value_type
  Tile<D, VGrid>::calculate_moment(
    runko::index_t x,
    runko::index_t y,
    runko::index_t z,
    MCF func,
    runko::index_t species)
{
  auto idx = std::array<runko::index_t, 3> { x, y, z };
  assert_inside(idx);
  const auto mds = containers_[species].mds();
  const auto w   = tyvi::mdgrid_work {};
  return mds[idx][].calculate_moment(w, func);
}

template<std::size_t D, VelGridType VGrid>
void
  Tile<D, VGrid>::debug_BC()
{
  // TODO: add other axes
  const auto extents                 = this->yee_lattice_.extents_wout_halo();
  const auto x_full                  = std::tuple { 0, 2 * halo_size + extents[0] };
  const auto y_full                  = std::tuple { 0, 2 * halo_size + extents[1] };
  [[maybe_unused]] const auto z_full = std::tuple { 0, 2 * halo_size + extents[2] };

  const auto w = tyvi::mdgrid_work {};

  for(auto& species: containers_) {
    // Create submdspans for the different halo regions and their accompanying
    // destination regions the submdspans' indices align so that the index of the point
    // in the halo region corresponds to the same index for the point in the destination
    // region

    const auto z_left_halo = std::submdspan(
      std::forward<decltype(species.mds())>(species.mds()),
      x_full,
      y_full,
      std::tuple { 0, halo_size });
    const auto z_left_dest = std::submdspan(
      std::forward<decltype(species.mds())>(species.mds()),
      x_full,
      y_full,
      std::tuple { extents[2], extents[2] + halo_size });
    const auto z_right_halo = std::submdspan(
      std::forward<decltype(species.mds())>(species.mds()),
      x_full,
      y_full,
      std::tuple { extents[2] + halo_size, extents[2] + 2 * halo_size });
    const auto z_right_dest = std::submdspan(
      std::forward<decltype(species.mds())>(species.mds()),
      x_full,
      y_full,
      std::tuple { halo_size, 2 * halo_size });


    for(auto idx: tyvi::sstd::index_space(z_left_halo)) {
      z_left_halo[idx][].recv_data(w, z_left_dest[idx][]);
    }

    for(auto idx: tyvi::sstd::index_space(z_right_halo)) {
      z_right_halo[idx][].recv_data(w, z_right_dest[idx][]);
    }
  }

  w.wait();
}

template<std::size_t D, VelGridType VGrid>
void
  Tile<D, VGrid>::deposit_current()
{
  const auto w = tyvi::mdgrid_work {};
  this->yee_lattice_.clear_current();
  auto J_grid       = runko::VecGrid<value_type>(containers_[0].extents());
  const auto J_smds = nonhalo_submds(J_grid.staging_mds());


  // The lambdas to calculate net velocity in the cell (1st moment)
  const auto vx_lambda = [] (double u_x, [[maybe_unused]] double u_y, [[maybe_unused]] double u_z, double gamma) { return u_x / gamma; };
  const auto vy_lambda = [] ([[maybe_unused]] double u_x, double u_y, [[maybe_unused]] double u_z, double gamma) { return u_y / gamma; };
  const auto vz_lambda = [] ([[maybe_unused]] double u_x, [[maybe_unused]] double u_y, double u_z, double gamma) { return u_z / gamma; };

  for(auto& species: containers_) {
    const auto nh_mds = nonhalo_submds(species.mds());

    // What we have to multiply by to get current from v
    const auto Jmult  = static_cast<value_type>(species.charge());

    // calculate the components of J for every cell
    for(auto idx: tyvi::sstd::index_space(J_smds)) {
      J_smds[idx][0] += nh_mds[idx][].calculate_moment(w, vx_lambda) * Jmult;
      J_smds[idx][1] += nh_mds[idx][].calculate_moment(w, vy_lambda) * Jmult;
      J_smds[idx][2] += nh_mds[idx][].calculate_moment(w, vz_lambda) * Jmult;
    }
  }
  w.sync_from_staging(J_grid).wait();
  this->yee_lattice_.deposit_current(J_grid);
}

template<std::size_t D, VelGridType VGrid>
void
  Tile<D, VGrid>::accelerate()
{

  // First we copy the E field to staging
  auto E_grid           = runko::VecGrid<value_type>(containers_[0].extents());
  const auto E_mds      = nonhalo_submds(this->yee_lattice_.mds_E());
  const auto E_grid_mds = nonhalo_submds(E_grid.mds());
  const auto w          = tyvi::mdgrid_work {};
  w.for_each_index(E_grid_mds, [=](const auto idx, const auto tidx) {
    E_grid_mds[idx][tidx] = E_mds[idx][tidx];
  });
  w.sync_to_staging(E_grid).wait();
  const auto E_smds = nonhalo_submds(E_grid.staging_mds());

  // lambda to accelerate along a single axis (needed to do async properly)
  // dt parameter allows partial steps (for Strang-splitting), 1.0 is a full step
  auto accelerate_axis =
  [&, this, E_smds](runko::index_t ax, value_type dt) {
      for(auto& species: this->containers_) {
        const auto nh_mds = nonhalo_submds(species.mds());
        const auto qpm    = static_cast<value_type>(species.charge() / species.mass());
        for(auto idx: tyvi::sstd::index_space(nh_mds)) {
          const auto a = qpm * E_smds[idx][ax];
          nh_mds[idx][].accelerate(w, a * dt, ax);
        }
      }
      w.wait();
      this->clean_up();
    };

  // accelerate along all the axes (TODO implement Strang-splitting)
  accelerate_axis(0, 1.0f);
  accelerate_axis(1, 1.0f);
  accelerate_axis(2, 1.0f);
}

template<std::size_t D, VelGridType VGrid>
std::vector<mpi4cpp::mpi::request>
  Tile<D, VGrid>::send_data(
    mpi4cpp::mpi::communicator& comm,
    const int dest,
    const int mode,
    const int tag)
{

// GPU backend requires GPU-aware MPI to pass device pointers directly;
// CPU backend uses host memory where standard MPI works.
#ifndef TYVI_BACKEND_CPU
  if(not toolbox::system_supports_gpu_aware_mpi()) {
    throw std::runtime_error { "GPU backend requires GPU-aware MPI." };
  }
#endif

  using runko::comm_mode;

  // Check which mode we are communicating in; only modes relevant for vlv::Tile are
  // processed here, rest are forwarded to emf::Tile
  switch(static_cast<comm_mode>(mode)) {
    case comm_mode::vlv_particle: {
      // Number of spatial cells (VlasovGrids) in a tile, also including halo regions
      const auto tot_spatial_cells = extents_[0] * extents_[1] * extents_[2];

      // Helper function to calculate the tag for MPI sends and recvs.
      // Tag must be unique for each idx in the spatial grid of a tile since the
      // communication is done one VlasovGrid at a time. Corgi uses tags to
      // differentiate different tiles etc. so we must offset the tags by the amount of
      // total spatial cells and different species
      auto get_vlv_tag = [this, tot_spatial_cells, tag]
        (std::array<runko::index_t, 3> idx, runko::index_t species) -> int {
        return runko::checked_cast<int>(
          static_cast<std::size_t>(tag * this->containers_.size()) * tot_spatial_cells +
          tot_spatial_cells * species + idx[0] * this->extents_[1] * this->extents_[2] +
          idx[1] * this->extents_[2] + idx[2]);
      };

      // Function to check whether or not the given index is inside the hollow grid we
      // need to communicate i.e. we skip the halo regions (they are only needed for
      // local comm) but also the very inside of the grid from where no halo region of a
      // neighboring tile will need data
      auto is_inside = [this](std::array<runko::index_t, 3> idx) -> bool {
        return (idx[0] >=      halo_size && idx[0] < this->extents_[0] -      halo_size &&
                idx[1] >=      halo_size && idx[1] < this->extents_[1] -      halo_size &&
                idx[2] >=      halo_size && idx[2] < this->extents_[2] -      halo_size)&&
              !(idx[0] >= 2u * halo_size && idx[0] < this->extents_[0] - 2u * halo_size &&
                idx[1] >= 2u * halo_size && idx[1] < this->extents_[1] - 2u * halo_size &&
                idx[2] >= 2u * halo_size && idx[2] < this->extents_[2] - 2u * halo_size);
      };

      // Create a list of requests since each spatial cell and species will have their own
      auto requests = std::vector<mpi4cpp::mpi::request>();

      // Loop through all species
      for(runko::index_t i = 0; i < containers_.size(); i++) {
        const auto mds = containers_[i].mds();

        // Loop through all spatial cells
        for(auto idx: tyvi::sstd::index_space(mds)) {
          const auto indices =
            std::array<runko::index_t, 3> { static_cast<runko::index_t>(idx[0]),
                                            static_cast<runko::index_t>(idx[1]),
                                            static_cast<runko::index_t>(idx[2]) };
          if(!is_inside(indices)) continue;
          const auto v_grid_span = mds[idx][].span();
          requests.push_back(comm.isend( // Create actual MPI send
            dest,
            get_vlv_tag(indices, i),
            v_grid_span.data(), 
            runko::checked_cast<int>(v_grid_span.size())
          ));
        }
      }
      return requests;
    }
    default:
      return emf::Tile<D>::send_data(comm, dest, mode, tag);  // Forward to emf::Tile
  }
}

template<std::size_t D, VelGridType VGrid>
std::vector<mpi4cpp::mpi::request>
  Tile<D, VGrid>::recv_data(
    mpi4cpp::mpi::communicator& comm,
    const int orig,
    const int mode,
    const int tag)
{
// GPU backend requires GPU-aware MPI to pass device pointers directly;
// CPU backend uses host memory where standard MPI works.
#ifndef TYVI_BACKEND_CPU
  if(not toolbox::system_supports_gpu_aware_mpi()) {
    throw std::runtime_error { "GPU backend requires GPU-aware MPI." };
  }
#endif

  using runko::comm_mode;

  // Check which mode we are communicating in; only modes relevant for vlv::Tile are
  // processed here, rest are forwarded to emf::Tile

  switch(static_cast<comm_mode>(mode)) {
    case comm_mode::vlv_particle: {
      // Number of spatial cells (VlasovGrids) in a tile, also including halo regions
      const auto tot_spatial_cells = extents_[0] * extents_[1] * extents_[2];

      // Helper function to calculate the tag for MPI sends and recvs.
      // Tag must be unique for each idx in the spatial grid of a tile since the
      // communication is done one VlasovGrid at a time. Corgi uses tags to
      // differentiate different tiles etc. so we must offset the tags by the amount of
      // total spatial cells and different species
      auto get_vlv_tag = [this, tot_spatial_cells, tag]
        (std::array<runko::index_t, 3> idx, runko::index_t species) -> int {
        return runko::checked_cast<int>(
          static_cast<std::size_t>(tag * this->containers_.size()) * tot_spatial_cells +
          tot_spatial_cells * species + idx[0] * this->extents_[1] * this->extents_[2] +
          idx[1] * this->extents_[2] + idx[2]);
      };

      // Function to check whether or not the given index is inside the hollow grid we
      // need to communicate i.e. we skip the halo regions (they are only needed for
      // local comm) but also the very inside of the grid from where no halo region of a
      // neighboring tile will need data
      auto is_inside = [this](std::array<runko::index_t, 3> idx) -> bool {
        return (idx[0] >=      halo_size && idx[0] < this->extents_[0] -      halo_size &&
                idx[1] >=      halo_size && idx[1] < this->extents_[1] -      halo_size &&
                idx[2] >=      halo_size && idx[2] < this->extents_[2] -      halo_size)&&
              !(idx[0] >= 2u * halo_size && idx[0] < this->extents_[0] - 2u * halo_size &&
                idx[1] >= 2u * halo_size && idx[1] < this->extents_[1] - 2u * halo_size &&
                idx[2] >= 2u * halo_size && idx[2] < this->extents_[2] - 2u * halo_size);
      };

      // Create a list of requests since each spatial cell and species will have their own
      auto requests = std::vector<mpi4cpp::mpi::request>();

      // Loop through all species
      for(runko::index_t i = 0; i < containers_.size(); i++) {
        const auto mds = containers_[i].mds();

        // Loop through all spatial cells
        for(auto idx: tyvi::sstd::index_space(mds)) {
          const auto indices =
            std::array<runko::index_t, 3> { static_cast<runko::index_t>(idx[0]),
                                            static_cast<runko::index_t>(idx[1]),
                                            static_cast<runko::index_t>(idx[2]) };
          if(!is_inside(indices)) continue;
          const auto v_grid_span = mds[idx][].span();
          requests.push_back(comm.irecv( // Create actual MPI recv
            orig,
            get_vlv_tag(indices, i),
            v_grid_span.data(), 
            runko::checked_cast<int>(v_grid_span.size())
          ));
        }
      }

      return requests;
    }
    default:
      return emf::Tile<D>::recv_data(comm, orig, mode, tag);  // Forward to emf::Tile
  }
}

template<std::size_t D, VelGridType VGrid>
void
  Tile<D, VGrid>::local_communication(
    const corgi::Tile<D>& other_base,
    const std::array<int, D> dir_to_other,
    const int mode)
{
  auto const* const other_base_ptr = &other_base;
  using runko::comm_mode;

  // First check if the communication is not something that needs vlv::Tile
  if(static_cast<comm_mode>(mode) != comm_mode::vlv_particle) {
    // If not, do the communication using emf::Tile
    emf::Tile<D>::local_communication(other_base, dir_to_other, mode);
    return;
  }

  // Cast to correct type of Tile, throw if it fails
  if(const auto* other = dynamic_cast<const Tile<D, VGrid>*>(other_base_ptr)) {
    switch(static_cast<comm_mode>(
      mode)) {  // Go through the relevant comm_modes (unnecessary for only one)
      case comm_mode::vlv_particle: {
        const auto w = tyvi::mdgrid_work {};

        // Update all VlasovMeshes in the subregion (on this Tile's halo region) from
        // the corresponding subregion on the other Tile (not on halo region).
        for(runko::index_t i = 0; i < containers_.size(); i++) {
          auto recv_mds =
            this->yee_lattice_.subregion(dir_to_other, containers_[i].mds());

          const auto send_mds = 
            other->yee_lattice_.corresponding_subregion(dir_to_other, other->containers_[i].mds());

          for(auto idx: tyvi::sstd::index_space(recv_mds)) {
            recv_mds[idx][].recv_data(w, send_mds[idx][]);
          }
        }
        w.wait();
        break;
      }
      default:
        throw std::logic_error { std::format(
          "vlv::Tile::local_communication does not support given "
          "communication mode: {}",
          mode) };
    }
  } else {
    throw std::runtime_error {
      "vlv::Tile::local_communication assumes that the other tile is "
      "vlv::Tile."
    };
  }
}

}  // namespace vlv

template class vlv::Tile<3, vlv::DenseGrid>;
