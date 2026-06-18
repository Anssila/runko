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
    )
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
void Tile<D, VGrid>::DebugAccelerate(runko::index_t x, runko::index_t y, runko::index_t z, double ax, double ay, double az, double dt){
    auto ax_ = static_cast<value_type>(ax);
    auto ay_ = static_cast<value_type>(ay);
    auto az_ = static_cast<value_type>(az);
    auto dt_ = static_cast<value_type>(dt);

    const auto mds = grid_.mds();
    auto idx = std::array<runko::index_t,3>{x + halo_size,y + halo_size,z + halo_size};
    mds[idx][].Shift(ax_, ay_, az_, dt_);
    
}

template<std::size_t D, VelGridType VGrid>
void Tile<D, VGrid>::SetVelGrid(runko::index_t x, runko::index_t y, runko::index_t z, VDF distribution){
    auto idx = std::array<runko::index_t,3>{x + halo_size,y + halo_size,z + halo_size};
    const auto mds = grid_.mds();
    mds[idx][].SetGridData(distribution);
}

template<std::size_t D, VelGridType VGrid>
VlasovGrid& Tile<D, VGrid>::GetVelGrid(runko::index_t x, runko::index_t y, runko::index_t z){
    auto idx = std::array<runko::index_t,3>{x + halo_size,y + halo_size,z + halo_size};
    const auto mds = grid_.mds();
    return mds[idx][];
}

template<std::size_t D, VelGridType VGrid>
void Tile<D, VGrid>::Translate(){ // TODO: other axes and strang-splitting
    const auto nh_mds = nonhalo_submds(grid_.mds());
    const auto mds = grid_.mds();
    const auto getInds = [=](uint64_t x, uint64_t y, uint64_t z){ // shifting the indices from nh_mds to mds
        return std::array<uint64_t,3>{ x + halo_size, y + halo_size, z + halo_size };
    };

    for (auto idx : tyvi::sstd::index_space(nh_mds)){
        auto neighbors = std::vector<VlasovGrid*>();
        neighbors.push_back(&mds[getInds(idx[0],idx[1],idx[2]-1)][]); // TODO allow higher order reconstruction / interpolation by adding more neighbors
        neighbors.push_back(&mds[getInds(idx[0],idx[1],idx[2]+0)][]);
        neighbors.push_back(&mds[getInds(idx[0],idx[1],idx[2]+1)][]);
        
        mds[getInds(idx[0],idx[1],idx[2])][].TranslateZ(neighbors, static_cast<value_type>(this->cfl_));
    }

    // TODO do communication here!! Or maybe split cleaning into a separate function?

    for (auto idx : tyvi::sstd::index_space(nh_mds)){
        mds[getInds(idx[0],idx[1],idx[2])][].Clean();
    }
}

} // namespace vlv

template class vlv::Tile<3, vlv::DenseGrid>;
