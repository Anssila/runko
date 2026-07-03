#include "vlasov_container.h"

namespace vlv {

template<VelGridType VGrid>
VlasovContainer<VGrid>::VlasovContainer(
    VlasovContainerArgs<VGrid> args) :
    charge_(args.charge),
    mass_(args.mass),
    grid_(
        args.spatial_extents[0],
        args.spatial_extents[1],
        args.spatial_extents[2]
    ) {

    // Initialize the spatial grid
    const auto mds = grid_.mds();
    auto Nvx = args.velocity_extents[0];
    auto Nvy = args.velocity_extents[1];
    auto Nvz = args.velocity_extents[2];
    const auto init_vals = args.u_init;
    const auto init_func = args.init_func;
    for (auto idx : tyvi::sstd::index_space(mds)){
        mds[idx][].set_size(Nvx, Nvy, Nvz);
        mds[idx][].InitZero();
        init_func(mds[idx][], init_vals);
    }
}


} // namespace vlv


template class vlv::VlasovContainer<vlv::DenseGrid>;