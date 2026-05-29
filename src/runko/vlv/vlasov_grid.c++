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

void DenseGrid::Shift_dir(value_type dv, size_t ax, [[maybe_unused]] const size_t order){
    auto grid = *grid_;
    auto new_grid = *new_grid_;

    // const auto s_grid = grid.staging_mds();
    // const auto s_new = new_grid.staging_mds();

    value_type shift_ind = dv / deltaU_[ax]; // how many indicies we shift by
    value_type min = std::floor(shift_ind); // the relative index of the cell with lower index we need to update
    value_type max = std::ceil(shift_ind); // the other relative index we need to update for every cell 

    [[maybe_unused]] value_type t_min = shift_ind - max; // TODO why
    [[maybe_unused]] value_type t_max = shift_ind - min; // TODO also does the interpolator work properly ?

    toolbox::Vec3 dir = {size_t{0},size_t{0},size_t{0}};
    dir[ax] = size_t{1};

    size_t min_ind = static_cast<size_t>(min);
    size_t max_ind = static_cast<size_t>(max);

    [[maybe_unused]] auto min_vec = dir * min_ind; // relative position of min
    [[maybe_unused]] auto max_vec = dir * max_ind; // relative position of max

    const auto w = tyvi::mdgrid_work{};

    w.sync_from_staging(grid).sync_from_staging(new_grid);

    // [[maybe_unused]] auto kernel = [TYVI_CMDS(grid, new_grid), this, ax, dv, order](const auto& idx) {
        
    //     toolbox::Vec3 id_vec = toolbox::Vec3(idx);
    //     auto interpolation_values = std::vector<value_type>();
    //     for (size_t i = 0; i < order+1; i++){
    //         std::array<size_t,3> inds = idx;
    //         inds[ax] += i - order/2;
    //         interpolation_values.push_back(grid_mds[inds][]);
    //     }

    //     auto float_dir = toolbox::Vec3(static_cast<value_type>(0.0f),static_cast<value_type>(0.0f),static_cast<value_type>(0.0f));
    //     float_dir[ax] = static_cast<value_type>(1.0f);

    //     auto v = toolbox::Vec3(GetVelFromIndex(idx));
    //     auto new_v = v + float_dir * dv;

    //     // we only need to consider two cells that this cell's fluid will end up in
    //     auto first_vec  = id_vec + dir * (this->GetIndexFromVel(dv, ax) + 0); 
    //     auto second_vec = id_vec + dir * (this->GetIndexFromVel(dv, ax) + 1); 

    //     auto first_ind = std::array<size_t,3>{first_vec[0],first_vec[1],first_vec[2]};
    //     auto second_ind = std::array<size_t,3>{second_vec[0],second_vec[1],second_vec[2]};

    //     // Create an interpolator that can be used to interpolate the values to the new positions 
    //     auto interp = [] (value_type t) {Interpolator(interpolation_values, t, order)}; // TODO should this be a proper object?

    //     new_grid_mds[first_ind][] = interp()

    //     for (int i = 0; i < 2; i++){ // we only need to loop over two cells for a linear interpolation scheme
    //         auto pos = start + dir * static_cast<toolbox::arithmetic auto>(i);
    //         std::array<size_t,3> inds = {pos[0], pos[1], pos[2]};
    //         value_type overlap = dv - this->GetVelFromIndex(pos[ax],ax);
    //         new_grid_mds[inds][] = grid_mds[idx][];
    //     }
    // };
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

vlv::VlasovGrid::value_type DenseGrid::GetTotalFluid() const{
    const auto staging_mds = grid_->staging_mds();
    value_type tot = static_cast<value_type>(0.0f);
    for (const auto idx : tyvi::sstd::index_space(staging_mds)) {
        tot += staging_mds[idx][];
    }
    return tot;
}

} // namespace vlv
