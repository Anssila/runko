#include "vlasov_grid.h"
#include "runko/tools/vector.h"
namespace vlv{

DenseGrid::DenseGrid(runko::index_t Nx, runko::index_t Ny, runko::index_t Nz){
    extents_ = {Nx, Ny, Nz};
    grid_ = std::make_unique<VelGrid>(Nx, Ny, Nz);
    new_grid_ = std::make_unique<VelGrid>(Nx, Ny, Nz);
};

runko::index_t DenseGrid::GetIndFromVel(value_type u, runko::index_t ax) const {
    value_type ind_val = u / deltaU_[ax] + static_cast<value_type>(extents_[ax]-1) * static_cast<value_type>(0.5f);
    return static_cast<runko::index_t>(ind_val);
}

vlv::VlasovGrid::value_type DenseGrid::GetVelFromInd(runko::index_t ind, runko::index_t ax) const {
    return (static_cast<value_type>(ind) - static_cast<value_type>(extents_[ax]-1)/2) * deltaU_[ax];
}

std::array<runko::index_t,3> DenseGrid::GetIndFromVel(std::array<value_type,3> u) const{
    return std::array<runko::index_t,3>{
        GetIndFromVel(u[0],0), 
        GetIndFromVel(u[1],1), 
        GetIndFromVel(u[2],2) 
    };
}

std::array<vlv::VlasovGrid::value_type,3> DenseGrid::GetVelFromInd(std::array<runko::index_t,3> inds) const{
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

inline void DenseGrid::ClampInds(std::array<int32_t,3> &inds, std::array<runko::index_t,3> extents){
    for (size_t ax = 0; ax < 3ul; ax++){
        int32_t val = inds[ax];
        int32_t max = extents[ax];
        inds[ax] = sstd::clamp(val,0,max-1);
    }
}

void DenseGrid::Shift_dir(value_type dv, runko::index_t ax, const runko::index_t order){
    auto& grid = *grid_;
    auto& new_grid = *new_grid_;

    value_type shift_ind = dv / deltaU_[ax]; // how many indicies we shift by
    value_type min = sstd::floor(shift_ind); // the relative index of the cell with lower index we need to update
    value_type max = sstd::ceil(shift_ind); // the other relative index we need to update for every cell 

    value_type t_min = min - shift_ind; 
    value_type t_max = max - shift_ind; 

    toolbox::Vec3 dir = toolbox::Vec3(int32_t{0},int32_t{0},int32_t{0});
    dir[ax] = int32_t{1};

    int32_t min_ind = static_cast<int32_t>(min);
    int32_t max_ind = static_cast<int32_t>(max);

    auto min_rel = dir * min_ind; // relative position of min
    auto max_rel = dir * max_ind; // relative position of max

    const auto w = tyvi::mdgrid_work{};

    std::array<runko::index_t,3> extents = extents_;

    auto kernel = [TYVI_CMDS(grid, new_grid), ax, min_rel, max_rel, t_min, t_max, order, extents, min_ind, max_ind] (const auto& idx) { // 
        
        toolbox::Vec3 ind_vec = toolbox::Vec3(static_cast<int32_t>(idx[0]),static_cast<int32_t>(idx[1]),static_cast<int32_t>(idx[2]));
        auto interpolation_values = std::vector<value_type>();
        for (int32_t i = 0; i < static_cast<int32_t>(order)+1; i++){
            std::array<int32_t,3> inds = {static_cast<int32_t>(idx[0]),static_cast<int32_t>(idx[1]),static_cast<int32_t>(idx[2])};
            inds[ax] += i - static_cast<int32_t>(order)/2;
            DenseGrid::ClampInds(inds, extents);
            interpolation_values.push_back(grid_mds[inds][]);
        }

        auto min_vec = ind_vec + min_rel;
        auto max_vec = ind_vec + max_rel;

        auto min_inds = std::array<int32_t,3>{min_vec[0], min_vec[1], min_vec[2]};
        auto max_inds = std::array<int32_t,3>{max_vec[0], max_vec[1], max_vec[2]};

        DenseGrid::ClampInds(min_inds, extents);
        DenseGrid::ClampInds(max_inds, extents);

        new_grid_mds[min_inds][] += Interpolator(interpolation_values, t_min, order); // TODO: Should Interpolator be an object instead?
        if (min_ind != max_ind) // Only add the second contribution if they are to different cells
            new_grid_mds[max_inds][] += Interpolator(interpolation_values, t_max, order);
    };

    w.for_each_index(new_grid, std::move(kernel));

    w.for_each_index(grid, [TYVI_CMDS(grid, new_grid)] (const auto& idx){ // Set grid to zeros in order to have a clean new_grid after the swap
        grid_mds[idx][] = static_cast<value_type>(0.0f);
    });

    w.wait();
    std::swap(grid_, new_grid_);
}

vlv::VlasovGrid::value_type DenseGrid::DebugGetFluid(std::array<runko::index_t,3> inds) const{ // This function is inefficient; don't use for anything important
    const auto w = tyvi::mdgrid_work{};
    w.sync_to_staging(*grid_).sync_to_staging(*new_grid_).wait();
    return grid_->staging_mds()[inds][];
}

std::vector<vlv::VlasovGrid::value_type> DenseGrid::DebugGetGrid() const{ // TODO: do this properly using the device buffer
    const auto staging_mds = grid_->staging_mds();
    std::vector<value_type> g;
    for (const auto idx : tyvi::sstd::index_space(staging_mds)){
        g.push_back(staging_mds[idx][]);
    }
    return g;
}

constexpr vlv::VlasovGrid::value_type DenseGrid::Interpolator(std::vector<value_type> &values, value_type t, runko::index_t order){

    // The interpolator takes in values describing the distribution around a center point (the middle value of "values" that
    // should always contain exactly 2*order+1 values) and interpolates the approximate distribution (to order "order") at t
    // that is measured relative to the center of the middle cell.

    switch (order)
    {
    case 0:
        // 0th order interpolation; distribution is modeled as a "staircase" and a simple linear interpolation will be performed
        return values[0] * (static_cast<value_type>(1.0f)-sstd::abs(t));
    default:
        std::stringstream msg;
        msg << "Dense grid interpolator of order " << order << " is not implemented!\n";
        throw std::runtime_error(msg.str());
    }
}


void DenseGrid::InitZero(){
    const auto staging_mds = grid_->staging_mds();
    for (const auto idx : tyvi::sstd::index_space(staging_mds)) {
        staging_mds[idx][] = static_cast<value_type>(0.0f);
    }
    const auto new_staging_mds = new_grid_->staging_mds();
    for (const auto idx : tyvi::sstd::index_space(new_staging_mds)) {
        new_staging_mds[idx][] = static_cast<value_type>(0.0f);
    }

    const auto w = tyvi::mdgrid_work{};
    w.sync_from_staging(*grid_).sync_from_staging(*new_grid_).wait();
}

void DenseGrid::InitDelta(std::array<value_type,3> v){
    const auto staging_mds = grid_->staging_mds();

    std::array<runko::index_t,3> v_inds = GetIndFromVel(v);
    
    for (const auto idx : tyvi::sstd::index_space(staging_mds)) {
        // If the index is the one corresponding to the given velocity, we set density to one, otherwise to zero
        staging_mds[idx][] = static_cast<value_type>(idx == v_inds ? 1.0f : 0.0f); 
    }

    const auto new_staging_mds = new_grid_->staging_mds();
    for (const auto idx : tyvi::sstd::index_space(new_staging_mds)) {
        new_staging_mds[idx][] = static_cast<value_type>(0.0f); // idx == v_inds ? 1.0f : 
    }

    const auto w = tyvi::mdgrid_work{};
    w.sync_from_staging(*grid_).sync_from_staging(*new_grid_).wait();
}


void DenseGrid::SetGridData(VlasovGrid::VelocityDistributionFunction distribution) {
    const auto staging_mds = grid_->staging_mds();
    
    for (const auto idx : tyvi::sstd::index_space(staging_mds)) {
        auto v = GetVelFromInd(idx);
        double x = static_cast<double>(v[0]);
        double y = static_cast<double>(v[1]);
        double z = static_cast<double>(v[2]);
        staging_mds[idx][] = static_cast<value_type>(distribution(x,y,z));
    }

    const auto w = tyvi::mdgrid_work{};
    w.sync_from_staging(*grid_).wait();
}


vlv::VlasovGrid::value_type DenseGrid::DebugGetTotalFluid() const{
    const auto w = tyvi::mdgrid_work{};
    w.sync_to_staging(*grid_).sync_to_staging(*new_grid_).wait();
    const auto staging_mds = grid_->staging_mds();
    value_type tot = static_cast<value_type>(0.0f);
    for (const auto idx : tyvi::sstd::index_space(staging_mds)) {
        tot += staging_mds[idx][];
    }
    return tot;
}

} // namespace vlv
