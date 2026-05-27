#include "vlasov_grid.h"

namespace vlv{

DenseGrid::DenseGrid(std::size_t Nx, std::size_t Ny, std::size_t Nz){
    extents_ = {Nx, Ny, Nz};
    grid_ = VelGrid(Nx, Ny, Nz);
};

void DenseGrid::Shift_x(value_type dv){
    dummy += dv;
}

void DenseGrid::Shift_y(value_type dv){
    dummy += dv;
}

void DenseGrid::Shift_z(value_type dv){
    dummy += dv;
}

void DenseGrid::InitZero(){
    const auto staging_mds = grid_.staging_mds();
    for (const auto idx : tyvi::sstd::index_space(staging_mds)) {
        staging_mds[idx][] = static_cast<value_type>(0.0f);
    }
}

vlv::VlasovGrid::value_type DenseGrid::GetTotalFluid(){
    const auto staging_mds = grid_.staging_mds();
    value_type tot = static_cast<value_type>(0.0f);
    for (const auto idx : tyvi::sstd::index_space(staging_mds)) {
        tot += staging_mds[idx][];
    }
    return tot;
}

} // namespace vlv