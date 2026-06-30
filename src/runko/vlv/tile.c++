#include "tile.h"

namespace vlv {

template<std::size_t D, VelGridType VGrid>
Tile<D, VGrid>::Tile(
    std::array<std::size_t, 3> tile_grid_indices, 
    const toolbox::ConfigParser& conf) :
    corgi::Tile<D>(),
    emf::Tile<D>(tile_grid_indices, conf), 
    grid_(
        static_cast<runko::index_t>(conf.get_or_throw<std::size_t>("NxMesh")) + 2 * halo_size,
        static_cast<runko::index_t>(conf.get_or_throw<std::size_t>("NyMesh")) + 2 * halo_size,
        static_cast<runko::index_t>(conf.get_or_throw<std::size_t>("NzMesh")) + 2 * halo_size
    ),
    extents_{
        static_cast<runko::index_t>(conf.get_or_throw<std::size_t>("NxMesh")) + 2 * halo_size,
        static_cast<runko::index_t>(conf.get_or_throw<std::size_t>("NyMesh")) + 2 * halo_size,
        static_cast<runko::index_t>(conf.get_or_throw<std::size_t>("NzMesh")) + 2 * halo_size
    }
    {
    auto deltaUx = conf.get<float>("deltaUx");
    auto deltaUy = conf.get<float>("deltaUy");
    auto deltaUz = conf.get<float>("deltaUz");
    auto x_infty = conf.get<float>("inftyx");
    auto y_infty = conf.get<float>("inftyy");
    auto z_infty = conf.get<float>("inftyz");

    auto initVelGrid = [=](VGrid &vel_grid){
        if      (deltaUx.has_value() && !x_infty.has_value()) vel_grid.SetDelta({deltaUx.value(), deltaUy.value_or(deltaUx.value()), deltaUz.value_or(deltaUx.value())}); 
        else if (!deltaUx.has_value() && x_infty.has_value()) vel_grid.SetInfty({x_infty.value(), y_infty.value_or(x_infty.value()), z_infty.value_or(x_infty.value())}); 
        else {
            std::stringstream msg;
            if (!deltaUx.has_value() && !x_infty.has_value())
                msg << "Cannot create tile without either deltaU or infty!\n";
            else
                msg << "Cannot create tile with both deltaU and infty!\n";
            throw std::runtime_error(msg.str());
        }
    };


    auto Nvx = static_cast<runko::index_t>(conf.get_or_throw<std::size_t>("Nvx"));
    auto Nvy = static_cast<runko::index_t>(conf.get_or_throw<std::size_t>("Nvy"));
    auto Nvz = static_cast<runko::index_t>(conf.get_or_throw<std::size_t>("Nvz"));

    const auto mds = grid_.mds();

    for (auto idx : tyvi::sstd::index_space(mds)){
        mds[idx][].SetSize(Nvx, Nvy, Nvz);
        mds[idx][].InitZero();
        initVelGrid(mds[idx][]);
    }
}

template<std::size_t D, VelGridType VGrid>
void Tile<D, VGrid>::AssertInside(std::array<runko::index_t, 3> idx) const{
    if (idx[0] >= extents_[0] || idx[1] >= extents_[1] || idx[2] >= extents_[2])
        throw std::runtime_error{ std::format("Indicies out of range! Got {}, {}, {} while extents are {}, {}, {}\n", idx[0], idx[1], idx[2], extents_[0], extents_[1], extents_[2]) };
}

template<std::size_t D, VelGridType VGrid>
void Tile<D, VGrid>::DebugAccelerate(runko::index_t x, runko::index_t y, runko::index_t z, double ax, double ay, double az, double dt){
    auto ax_ = static_cast<value_type>(ax);
    auto ay_ = static_cast<value_type>(ay);
    auto az_ = static_cast<value_type>(az);
    auto dt_ = static_cast<value_type>(dt);

    const auto mds = grid_.mds();
    auto idx = std::array<runko::index_t,3>{x,y,z};
    AssertInside(idx); 
    const auto w = tyvi::mdgrid_work{};
    mds[idx][].Shift(w, ax_, ay_, az_, dt_);
    w.wait();
}

template<std::size_t D, VelGridType VGrid>
void Tile<D, VGrid>::SetVelGrid(runko::index_t x, runko::index_t y, runko::index_t z, VDF distribution){
    auto idx = std::array<runko::index_t,3>{x,y,z};
    AssertInside(idx); 
    const auto mds = grid_.mds();
    mds[idx][].SetGridData(distribution);
}

template<std::size_t D, VelGridType VGrid>
VlasovGrid& Tile<D, VGrid>::GetVelGrid(runko::index_t x, runko::index_t y, runko::index_t z){
    auto idx = std::array<runko::index_t,3>{x,y,z};
    AssertInside(idx); 
    const auto mds = grid_.mds();
    return mds[idx][];
}

template<std::size_t D, VelGridType VGrid>
void Tile<D, VGrid>::Translate(){ 

    // TODO: other axes and strang-splitting
    // TODO: make sure iteration includes halo regions of the other axes while translating along some axis

    const auto nh_mds = nonhalo_submds(grid_.mds());
    const auto mds = grid_.mds();
    const auto getInds = [=](uint64_t x, uint64_t y, uint64_t z){ // shifting the indices from nh_mds to mds
        return std::array<uint64_t,3>{ x + halo_size, y + halo_size, z + halo_size };
    };

    const auto w = tyvi::mdgrid_work{};

    for (auto idx : tyvi::sstd::index_space(nh_mds)){
        auto neighbors = std::vector<VlasovGrid*>();
        neighbors.push_back(&mds[getInds(idx[0],idx[1],idx[2]-1)][]); // TODO allow higher order reconstruction / interpolation by adding more neighbors
        neighbors.push_back(&mds[getInds(idx[0],idx[1],idx[2]+0)][]);
        neighbors.push_back(&mds[getInds(idx[0],idx[1],idx[2]+1)][]);

        mds[getInds(idx[0],idx[1],idx[2])][].TranslateZ(w, neighbors, static_cast<value_type>(this->cfl_));
    }
    w.wait();
}

template<std::size_t D, VelGridType VGrid>
void Tile<D, VGrid>::CleanUp(){
    const auto mds = grid_.mds();
    const auto w = tyvi::mdgrid_work{};

    for (auto idx : tyvi::sstd::index_space(mds)){
        mds[idx][].Clean(w);
    }
    w.wait();
}

template<std::size_t D, VelGridType VGrid>
Tile<D, VGrid>::value_type Tile<D, VGrid>::CalculateMoment(runko::index_t x, runko::index_t y, runko::index_t z, MCF func){
    auto idx = std::array<runko::index_t,3>{x,y,z};
    AssertInside(idx); 
    const auto mds = grid_.mds();
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

    // Create submdspans for the different halo regions and their accompanying destination regions 
    // the submdspans' indices align so that the index of the point in the halo region corresponds to
    // the same index for the point in the destination region

    const auto z_left_halo = std::submdspan( std::forward<decltype(grid_.mds())>(grid_.mds()), x_full, y_full, std::tuple { 0, halo_size } );
    const auto z_left_dest = std::submdspan( std::forward<decltype(grid_.mds())>(grid_.mds()), x_full, y_full, std::tuple { extents[2], extents[2] + halo_size } );
    const auto z_right_halo = std::submdspan( std::forward<decltype(grid_.mds())>(grid_.mds()), x_full, y_full, std::tuple { extents[2] + halo_size , extents[2] + 2 * halo_size } );
    const auto z_right_dest = std::submdspan( std::forward<decltype(grid_.mds())>(grid_.mds()), x_full, y_full, std::tuple { halo_size , 2 * halo_size } );

    const auto w = tyvi::mdgrid_work{};

    for (auto idx : tyvi::sstd::index_space(z_left_halo)){
        z_left_halo[idx][].SendData(w,z_left_dest[idx][]);
    }

    for (auto idx : tyvi::sstd::index_space(z_right_halo)){
        z_right_halo[idx][].SendData(w,z_right_dest[idx][]);
    }
    w.wait();
}

template<std::size_t D, VelGridType VGrid>
void Tile<D, VGrid>::deposit_current(){
    const auto nh_mds = nonhalo_submds(grid_.mds());
    this->yee_lattice_.clear_current();
    auto J_grid = runko::VecGrid<value_type>(grid_.grid_extents());
    const auto J_smds = J_grid.staging_mds();
    const auto w = tyvi::mdgrid_work{};

    const auto Jx_lambda = [] ([[maybe_unused]] double x, [[maybe_unused]] double y, [[maybe_unused]] double z, double gamma) { return x / gamma; };
    const auto Jy_lambda = [] ([[maybe_unused]] double x, [[maybe_unused]] double y, [[maybe_unused]] double z, double gamma) { return y / gamma; };
    const auto Jz_lambda = [] ([[maybe_unused]] double x, [[maybe_unused]] double y, [[maybe_unused]] double z, double gamma) { return z / gamma; };

    for (auto idx : tyvi::sstd::index_space(nh_mds)){
        J_smds[idx][0] = nh_mds[idx][].CalculateMoment(w, Jx_lambda);
        J_smds[idx][1] = nh_mds[idx][].CalculateMoment(w, Jy_lambda);
        J_smds[idx][2] = nh_mds[idx][].CalculateMoment(w, Jz_lambda);
    }
    w.wait();
    w.sync_from_staging(J_grid).wait(); // TODO: when to wait??

    this->yee_lattice_.deposit_current(J_grid);
}

} // namespace vlv

template class vlv::Tile<3, vlv::DenseGrid>;
