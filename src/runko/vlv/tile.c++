#include "tile.h"

namespace vlv {

template<std::size_t D, VelGridType VGrid>
Tile<D, VGrid>::Tile(
    std::array<std::size_t, 3> tile_grid_indices, 
    const toolbox::ConfigParser& conf) :
    corgi::Tile<D>(),
    emf::Tile<D>(tile_grid_indices, conf),
    containers_(),
    extents_{
        static_cast<runko::index_t>(conf.get_or_throw<std::vector<std::ptrdiff_t>>("n_cells_per_tile")[0]) + 2 * halo_size,
        static_cast<runko::index_t>(conf.get_or_throw<std::vector<std::ptrdiff_t>>("n_cells_per_tile")[1]) + 2 * halo_size,
        static_cast<runko::index_t>(conf.get_or_throw<std::vector<std::ptrdiff_t>>("n_cells_per_tile")[2]) + 2 * halo_size
    }
    {

    using opt_args_t = std::optional<vlv::VlasovContainerArgs<VGrid>>;

    // Converter function to make sure the values given from Python are converted correctly
    const auto converter = [] (std::vector<double> v) {
        return std::array<vlv::VlasovGrid::value_type,3>{
            static_cast<vlv::VlasovGrid::value_type>(v[0]),
            static_cast<vlv::VlasovGrid::value_type>(v[1]),
            static_cast<vlv::VlasovGrid::value_type>(v[2])
        };
    };

    // Return optional containing the arguments if q and m are found in conf.
    const auto make_opt_args =
        [&](const std::string& q_label, const std::string& m_label) -> opt_args_t {

        const auto q_to_qm_tuple = [&](const auto q) {
            return conf.get<double>(m_label).transform(
            [&](const auto m) { return std::tuple { q, m }; } );
        };

        const auto qm_tuple_to_args = [&](const auto qm) {
            const auto [q, m] = qm;
            return vlv::VlasovContainerArgs<VGrid> {
                .charge = q,
                .mass = m,
                .spatial_extents = {0,0,0},
                .velocity_extents = {0,0,0},
                .u_init = {0.0,0.0,0.0},
                .init_func = &VlasovGrid::set_u_max
            };
        };

        return conf.get<double>(q_label)
            .and_then(q_to_qm_tuple)
            .transform(qm_tuple_to_args);
    };

    const auto vel_exs = conf.get_or_throw<std::vector<std::ptrdiff_t>>("v_grid_extents");
    auto Nvx = static_cast<runko::index_t>(vel_exs[0]);
    auto Nvy = static_cast<runko::index_t>(vel_exs[1]);
    auto Nvz = static_cast<runko::index_t>(vel_exs[2]);

    // We get either u_max or u_res and deduce the other based on the extents
    auto u_max = conf.get<std::vector<double>>("u_max");
    auto u_res = conf.get<std::vector<double>>("u_res");

    // Based on which we get, we must pass the correct values and the correct function to the container
    std::function<void(VGrid&, std::array<vlv::VlasovGrid::value_type,3>)> init_func; // The function to call to init the VlasovGrid
    std::array<value_type,3> init_values; // The values to pass to the init_func

    if (u_res.has_value() && !u_max.has_value()) {
        init_func = &VGrid::set_u_res;
        init_values = converter(u_res.value());
    }
    else if (!u_res.has_value() && u_max.has_value()) {
        init_func = &VGrid::set_u_max;
        init_values = converter(u_max.value());
    }
    else {
        std::stringstream msg;
        if (!u_res.has_value() && !u_max.has_value())
            msg << "Cannot create tile without either u_res or u_max!\n";
        else
            msg << "Cannot create tile with both u_res and u_max!\n";
        throw std::runtime_error(msg.str());
    }

    for(auto i = 0uz; true; ++i) {
        const auto q_label = std::format("q{}", i);
        const auto m_label = std::format("m{}", i);

        if(const auto vcontainer_args = make_opt_args(q_label, m_label)) {
            auto args = vcontainer_args.value();
            args.spatial_extents = extents_;
            args.velocity_extents = {Nvx, Nvy, Nvz};
            args.u_init = init_values;
            args.init_func = init_func;
            containers_.emplace_back(vlv::VlasovContainer<VGrid> { args });
        } else {
            break;
        }
    }
}

template<std::size_t D, VelGridType VGrid>
void Tile<D, VGrid>::AssertInside(std::array<runko::index_t, 3> idx) const{
    if (idx[0] >= extents_[0] || idx[1] >= extents_[1] || idx[2] >= extents_[2])
        throw std::runtime_error{ std::format("Indicies out of range! Got {}, {}, {} while extents are {}, {}, {}\n", idx[0], idx[1], idx[2], extents_[0], extents_[1], extents_[2]) };
}

template<std::size_t D, VelGridType VGrid>
void Tile<D, VGrid>::DebugAccelerate(runko::index_t x, runko::index_t y, runko::index_t z, double ax, double ay, double az){
    auto ax_ = static_cast<value_type>(ax);
    auto ay_ = static_cast<value_type>(ay);
    auto az_ = static_cast<value_type>(az);
    const auto w = tyvi::mdgrid_work{};

    for (auto& species : containers_){
        const auto mds = species.mds();
        auto idx = std::array<runko::index_t,3>{x,y,z};
        AssertInside(idx); 
        mds[idx][].Shift(w, ax_, ay_, az_);
    }
    w.wait();
}

template<std::size_t D, VelGridType VGrid>
void Tile<D, VGrid>::SetVelGrid(runko::index_t x, runko::index_t y, runko::index_t z, VDF distribution, runko::index_t species){
    auto idx = std::array<runko::index_t,3>{x,y,z};
    AssertInside(idx); 
    const auto mds = containers_[species].mds();
    mds[idx][].SetGridData(distribution);
}

template<std::size_t D, VelGridType VGrid>
VlasovGrid& Tile<D, VGrid>::GetVelGrid(runko::index_t x, runko::index_t y, runko::index_t z, runko::index_t species){
    auto idx = std::array<runko::index_t,3>{x,y,z};
    AssertInside(idx); 
    const auto mds = containers_[species].mds();
    return mds[idx][];
}

template<std::size_t D, VelGridType VGrid>
void Tile<D, VGrid>::Translate(){ 

    // TODO: other axes and strang-splitting
    // TODO: make sure iteration includes halo regions of the other axes while translating along some axis
    const auto w = tyvi::mdgrid_work{};

    for (auto& species : containers_) {
        const auto mds = species.mds();
        const auto nh_mds = nonhalo_submds(mds);
        const auto getInds = [=](uint64_t x, uint64_t y, uint64_t z){ // shifting the indices from nh_mds to mds
            return std::array<uint64_t,3>{ x + halo_size, y + halo_size, z + halo_size };
        };


        for (auto idx : tyvi::sstd::index_space(nh_mds)){
            auto neighbors = std::vector<VlasovGrid*>();
            neighbors.push_back(&mds[getInds(idx[0],idx[1],idx[2]-1)][]); // TODO allow higher order reconstruction / interpolation by adding more neighbors
            neighbors.push_back(&mds[getInds(idx[0],idx[1],idx[2]+0)][]);
            neighbors.push_back(&mds[getInds(idx[0],idx[1],idx[2]+1)][]);

            mds[getInds(idx[0],idx[1],idx[2])][].TranslateZ(w, neighbors, static_cast<value_type>(this->cfl_));
        }
    }
    w.wait();
}

template<std::size_t D, VelGridType VGrid>
void Tile<D, VGrid>::CleanUp(){
    const auto w = tyvi::mdgrid_work{};
    for (auto& species : containers_){
        const auto mds = species.mds();
        for (auto idx : tyvi::sstd::index_space(mds)){
            mds[idx][].Clean(w);
        }
    }
    w.wait();
}

template<std::size_t D, VelGridType VGrid>
Tile<D, VGrid>::value_type Tile<D, VGrid>::CalculateMoment(runko::index_t x, runko::index_t y, runko::index_t z, MCF func, runko::index_t species){
    auto idx = std::array<runko::index_t,3>{x,y,z};
    AssertInside(idx); 
    const auto mds = containers_[species].mds();
    const auto w = tyvi::mdgrid_work{};
    return mds[idx][].CalculateMoment(w, func);
}

template<std::size_t D, VelGridType VGrid>
void Tile<D, VGrid>::DebugBC(){
    // TODO: add other axes
    const auto extents = this->yee_lattice_.extents_wout_halo();
    const auto x_full = std::tuple { 0, 2 * halo_size + extents[0] };
    const auto y_full = std::tuple { 0, 2 * halo_size + extents[1] };
    [[maybe_unused]] const auto z_full = std::tuple { 0, 2 * halo_size + extents[2] };

    const auto w = tyvi::mdgrid_work{};

    for (auto& species : containers_){
        // Create submdspans for the different halo regions and their accompanying destination regions 
        // the submdspans' indices align so that the index of the point in the halo region corresponds to
        // the same index for the point in the destination region

        const auto z_left_halo = std::submdspan( std::forward<decltype(species.mds())>(species.mds()), x_full, y_full, std::tuple { 0, halo_size } );
        const auto z_left_dest = std::submdspan( std::forward<decltype(species.mds())>(species.mds()), x_full, y_full, std::tuple { extents[2], extents[2] + halo_size } );
        const auto z_right_halo = std::submdspan( std::forward<decltype(species.mds())>(species.mds()), x_full, y_full, std::tuple { extents[2] + halo_size , extents[2] + 2 * halo_size } );
        const auto z_right_dest = std::submdspan( std::forward<decltype(species.mds())>(species.mds()), x_full, y_full, std::tuple { halo_size , 2 * halo_size } );


        for (auto idx : tyvi::sstd::index_space(z_left_halo)){
            z_left_halo[idx][].SendData(w,z_left_dest[idx][]);
        }

        for (auto idx : tyvi::sstd::index_space(z_right_halo)){
            z_right_halo[idx][].SendData(w,z_right_dest[idx][]);
        }
    }

    w.wait();
}

template<std::size_t D, VelGridType VGrid>
void Tile<D, VGrid>::deposit_current(){
    const auto w = tyvi::mdgrid_work{};
    this->yee_lattice_.clear_current();
    auto J_grid = runko::VecGrid<value_type>(containers_[0].extents());
    const auto J_smds = nonhalo_submds(J_grid.staging_mds());

    const auto vx_lambda = [] ([[maybe_unused]] double u_x, [[maybe_unused]] double u_y, [[maybe_unused]] double u_z, double gamma) { return u_x / gamma; };
    const auto vy_lambda = [] ([[maybe_unused]] double u_x, [[maybe_unused]] double u_y, [[maybe_unused]] double u_z, double gamma) { return u_y / gamma; };
    const auto vz_lambda = [] ([[maybe_unused]] double u_x, [[maybe_unused]] double u_y, [[maybe_unused]] double u_z, double gamma) { return u_z / gamma; };

    for (auto& species : containers_){
        const auto nh_mds = nonhalo_submds(species.mds());
        const auto Jmult = static_cast<value_type>(species.charge() * this->cfl_); // What we have to multiply by to get current from v

        for (auto idx : tyvi::sstd::index_space(J_smds)){
            J_smds[idx][0] += nh_mds[idx][].CalculateMoment(w, vx_lambda) * Jmult;
            J_smds[idx][1] += nh_mds[idx][].CalculateMoment(w, vy_lambda) * Jmult;
            J_smds[idx][2] += nh_mds[idx][].CalculateMoment(w, vz_lambda) * Jmult;
        }
    }
    w.sync_from_staging(J_grid).wait();
    this->yee_lattice_.deposit_current(J_grid);
}

template<std::size_t D, VelGridType VGrid>
void Tile<D, VGrid>::accelerate(){
    auto E_grid = runko::VecGrid<value_type>(containers_[0].extents());
    const auto E_mds = nonhalo_submds(this->yee_lattice_.mds_E());
    const auto E_grid_mds = nonhalo_submds(E_grid.mds());
    const auto w = tyvi::mdgrid_work{};
    w.for_each_index(E_grid_mds, [=] (const auto idx, const auto tidx) {
        E_grid_mds[idx][tidx] = E_mds[idx][tidx];
    });
    w.sync_to_staging(E_grid).wait();
    const auto E_smds = nonhalo_submds(E_grid.staging_mds());
    for (auto& species : containers_){
        const auto nh_mds = nonhalo_submds(species.mds());
        const auto qpm = static_cast<value_type>(species.charge() / species.mass());
        for (auto idx : tyvi::sstd::index_space(nh_mds)){
            const auto a_x = qpm * E_smds[idx][0];
            const auto a_y = qpm * E_smds[idx][1];
            const auto a_z = qpm * E_smds[idx][2];
            nh_mds[idx][].Shift(w, a_x, a_y, a_z);
        }
    }
    w.wait();
}

} // namespace vlv

template class vlv::Tile<3, vlv::DenseGrid>;
