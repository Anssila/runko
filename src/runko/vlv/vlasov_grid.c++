#include "vlasov_grid.h"
#include "runko/tools/vector.h"
namespace vlv{

DenseGrid::DenseGrid(std::size_t Nx, std::size_t Ny, std::size_t Nz){
    extents_ = {Nx, Ny, Nz};
    grid_ = new VelGrid(Nx, Ny, Nz);
    new_grid_ = new VelGrid(Nx, Ny, Nz);
};

size_t DenseGrid::GetIndexFromVel(value_type u, size_t ax) const {
    value_type ind_val = u / deltaU_[ax] + static_cast<value_type>(extents_[ax]-1) * static_cast<value_type>(0.5f);
    return static_cast<size_t>(ind_val);
}

vlv::VlasovGrid::value_type DenseGrid::GetVelFromIndex(size_t ind, size_t ax) const {
    return -infty_[ax] + static_cast<value_type>(ind) * deltaU_[ax];
}

void DenseGrid::SetDelta(std::array<value_type,3> deltas){
    deltaU_ = deltas;
    for (int i = 0; i < 3; i++){
        infty_[i] = static_cast<value_type>(extents_[i]-1) * static_cast<value_type>(0.5f) * deltaU_[i];
    }
}

void DenseGrid::SetInfty(std::array<value_type,3> inftys){
    infty_ = inftys;
    for (int i = 0; i < 3; i++){
        deltaU_[i] = infty_[i] / static_cast<value_type>(extents_[i]-1) * static_cast<value_type>(2.0f);
    }
}

void DenseGrid::Shift_dir(value_type dv, size_t ax){
    auto grid = *grid_;
    auto new_grid = *new_grid_;

    const auto s_grid = grid.staging_mds();
    const auto s_new = new_grid.staging_mds();

    const auto w = tyvi::mdgrid_work{};

    w.sync_from_staging(s_grid).sync_from_staging(s_new);

    auto kernel = [TYVI_CMDS(grid, new_grid)](const auto& idx) {
        toolbox::Vec3 dir = {size_t{0},size_t{0},size_t{0}};
        dir[ax] = size_t{1};
        toolbox::Vec3 id_vec = {idx[0],idx[1],idx[2]};
        auto start = id_vec + dir * (GetIndexFromVel(dv, ax) + 0); 

        for (int i = 0; i < 2; i++){
            
        }

        new_grid_mds[start] = grid_mds[start];
    };
}


void DenseGrid::InitZero(){
    const auto staging_mds = grid_->staging_mds();
    for (const auto idx : tyvi::sstd::index_space(staging_mds)) {
        staging_mds[idx][] = static_cast<value_type>(0.0f);
    }
}

vlv::VlasovGrid::value_type DenseGrid::GetTotalFluid() const{
    const auto staging_mds = grid_->staging_mds();
    value_type tot = static_cast<value_type>(0.0f);
    for (const auto idx : tyvi::sstd::index_space(staging_mds)) {
        tot += staging_mds[idx][];
    }
    return tot;
}

} // namespace vlv
