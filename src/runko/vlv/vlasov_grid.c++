#include "vlasov_grid.h"
#include "runko/tools/vector.h"
namespace vlv{

DenseGrid::DenseGrid(std::size_t Nx, std::size_t Ny, std::size_t Nz){
    extents_ = {Nx, Ny, Nz};
    grid_ = new VelGrid(Nx, Ny, Nz);
    new_grid_ = new VelGrid(Nx, Ny, Nz);
};

size_t DenseGrid::GetIndFromVel(value_type u, size_t ax) const {
    value_type ind_val = u / deltaU_[ax] + static_cast<value_type>(extents_[ax]-1) * static_cast<value_type>(0.5f);
    return static_cast<size_t>(ind_val);
}

vlv::VlasovGrid::value_type DenseGrid::GetVelFromInd(size_t ind, size_t ax) const {
    return -infty_[ax] + static_cast<value_type>(ind) * deltaU_[ax];
}

std::array<size_t,3> DenseGrid::GetIndFromVel(std::array<value_type,3> u) const{
    return std::array<size_t,3>{
        GetIndFromVel(u[0],0), 
        GetIndFromVel(u[1],1), 
        GetIndFromVel(u[2],2) 
    };
}

std::array<vlv::VlasovGrid::value_type,3> DenseGrid::GetVelFromInd(std::array<size_t,3> inds) const{
    return std::array<value_type,3>{
        GetVelFromInd(inds[0],0),
        GetVelFromInd(inds[1],1),
        GetVelFromInd(inds[2],2)
    };
}

void DenseGrid::SetDelta(std::array<value_type,3> deltas){
    deltaU_ = deltas;
    for (size_t i = 0; i < 3ul; i++){
        infty_[i] = static_cast<value_type>(extents_[i]-1ul) * static_cast<value_type>(0.5f) * deltaU_[i];
    }
}

void DenseGrid::SetInfty(std::array<value_type,3> inftys){
    infty_ = inftys;
    for (size_t i = 0; i < 3ul; i++){
        deltaU_[i] = infty_[i] / static_cast<value_type>(extents_[i]-1ul) * static_cast<value_type>(2.0f);
    }
}

inline void DenseGrid::ClampInds(std::array<size_t,3> &inds, std::array<size_t,3> ex){
    for (size_t ax = 0; ax < 3ul; ax++){
        inds[ax] = std::max(std::min(inds[ax],ex[ax]-size_t{1}),size_t{0});
    }
}

void DenseGrid::Shift_dir(value_type dv, size_t ax, [[maybe_unused]] const size_t order){
    auto grid = *grid_;
    auto new_grid = *new_grid_;

    // const auto s_grid = grid.staging_mds();
    // const auto s_new = new_grid.staging_mds();

    value_type shift_ind = dv / deltaU_[ax]; // how many indicies we shift by
    value_type min = std::floor(shift_ind); // the relative index of the cell with lower index we need to update
    value_type max = std::ceil(shift_ind); // the other relative index we need to update for every cell 

    value_type t_min = shift_ind - max; // TODO why
    value_type t_max = shift_ind - min; // TODO also does the interpolator work properly ?

    toolbox::Vec3 dir = toolbox::Vec3(size_t{0},size_t{0},size_t{0});
    dir[ax] = size_t{1};

    size_t min_ind = static_cast<size_t>(min);
    size_t max_ind = static_cast<size_t>(max);

    auto min_rel = dir * min_ind; // relative position of min
    auto max_rel = dir * max_ind; // relative position of max

    const auto w = tyvi::mdgrid_work{};

    w.sync_from_staging(grid).sync_from_staging(new_grid);

    std::array<size_t,3> ex = extents_;

    auto kernel = [TYVI_CMDS(grid, new_grid), ax, min_rel, max_rel, t_min, t_max, order, ex](const auto& idx) {
        
        toolbox::Vec3 ind_vec = toolbox::Vec3(static_cast<size_t>(idx[0]),static_cast<size_t>(idx[1]),static_cast<size_t>(idx[2]));
        auto interpolation_values = std::vector<value_type>();
        for (size_t i = 0; i < order+1; i++){
            std::array<size_t,3> inds = {idx[0],idx[1],idx[2]};
            inds[ax] += i - order/2;
            DenseGrid::ClampInds(inds, ex);
            interpolation_values.push_back(grid_mds[inds][]);
        }

        auto min_vec = ind_vec + min_rel;
        auto max_vec = ind_vec + max_rel;

        auto min_inds = std::array<size_t,3>{min_vec[0], min_vec[1], min_vec[2]};
        auto max_inds = std::array<size_t,3>{max_vec[0], max_vec[1], max_vec[2]};

        DenseGrid::ClampInds(min_inds, ex);
        DenseGrid::ClampInds(max_inds, ex);

        new_grid_mds[min_inds][] = Interpolator(interpolation_values, t_min, order); // TODO: Should Interpolator be an object instead?
        new_grid_mds[max_inds][] = Interpolator(interpolation_values, t_max, order);
    };

    w.for_each_index(new_grid, kernel).sync_to_staging(new_grid).wait();
}

constexpr vlv::VlasovGrid::value_type DenseGrid::Interpolator(std::vector<value_type> &values, value_type t, size_t order){

    // The interpolator takes in values describing the distribution around a center point (the middle value of "values" that
    // should always contain exactly 2*order+1 values) and interpolates the approximate distribution (to order "order") at t
    // that is measured relative to the center of the middle cell.

    switch (order)
    {
    case 0:
        // 0th order interpolation; distribution is modeled as a "staircase" and a simple linear interpolation will be performed
        return t < static_cast<value_type>(0.0f) ? values[0] * (static_cast<value_type>(1.0f) + t) : values[0] * t;
    default:
        std::stringstream msg;
        msg << "Interpolator of order " << order << " is not implemented!\n";
        throw std::runtime_error(msg.str());
    }
}


void DenseGrid::InitZero(){
    const auto staging_mds = grid_->staging_mds();
    for (const auto idx : tyvi::sstd::index_space(staging_mds)) {
        staging_mds[idx][] = static_cast<value_type>(0.0f);
    }
}

vlv::VlasovGrid::value_type DenseGrid::GetTotalFluid() const{ // TODO: Should this use the device??
    const auto staging_mds = grid_->staging_mds();
    value_type tot = static_cast<value_type>(0.0f);
    for (const auto idx : tyvi::sstd::index_space(staging_mds)) {
        tot += staging_mds[idx][];
    }
    return tot;
}

} // namespace vlv
