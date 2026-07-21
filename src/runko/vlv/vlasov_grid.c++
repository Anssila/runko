#include "vlasov_grid.h"
#include "runko/tools/vector.h"
#include "thrust/execution_policy.h"
#include "thrust/host_vector.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/reduce.h"
namespace vlv{

void DenseGrid::set_size(runko::index_t Nx, runko::index_t Ny, runko::index_t Nz){
    extents_ = {Nx, Ny, Nz};
    grid_ = std::make_unique<VelGrid>(Nx, Ny, Nz);
    new_grid_ = std::make_unique<VelGrid>(Nx, Ny, Nz);
};

constexpr runko::index_t DenseGrid::GetIndFromVel(value_type u, runko::index_t ex, value_type deltaU) {
    value_type ind_val = u / deltaU + static_cast<value_type>(ex-1) * static_cast<value_type>(0.5f);
    return static_cast<runko::index_t>(ind_val);
}

constexpr vlv::VlasovGrid::value_type DenseGrid::GetVelFromInd(runko::index_t ind, runko::index_t ex, value_type deltaU) {
    return (static_cast<value_type>(ind) - static_cast<value_type>(ex-1)/2) * deltaU;
}

std::array<runko::index_t,3> DenseGrid::GetIndFromVel(std::array<value_type,3> u) const{
    return std::array<runko::index_t,3>{
        GetIndFromVel(u[0], extents_[0], u_res_[0]), 
        GetIndFromVel(u[1], extents_[1], u_res_[1]), 
        GetIndFromVel(u[2], extents_[2], u_res_[2]) 
    };
}

std::array<vlv::VlasovGrid::value_type,3> DenseGrid::GetVelFromInd(std::array<runko::index_t,3> inds) const{
    return std::array<value_type,3>{
        GetVelFromInd(inds[0], extents_[0], u_res_[0]),
        GetVelFromInd(inds[1], extents_[1], u_res_[1]),
        GetVelFromInd(inds[2], extents_[2], u_res_[2])
    };
}

void DenseGrid::set_u_res(std::array<value_type,3> res){
    u_res_ = res;
    for (size_t i = 0; i < 3ul; i++){
        u_max_[i] = static_cast<value_type>(extents_[i]-1ul) * static_cast<value_type>(0.5f) * u_res_[i];
    }
}

void DenseGrid::set_u_max(std::array<value_type,3> max){
    u_max_ = max;
    for (size_t i = 0; i < 3ul; i++){
        u_res_[i] = u_max_[i] / static_cast<value_type>(extents_[i]-1ul) * static_cast<value_type>(2.0f);
    }
}

inline void DenseGrid::ClampInds(std::array<int32_t,3> &inds, std::array<runko::index_t,3> extents){
    for (size_t ax = 0; ax < 3ul; ax++){
        int32_t val = inds[ax];
        int32_t max = extents[ax];
        inds[ax] = sstd::clamp(val,0,max-1);
    }
}

void DenseGrid::Shift_dir(const tyvi::mdgrid_work& w, value_type dv, runko::index_t ax, const runko::index_t order){
    value_type shift_ind = dv / u_res_[ax]; // how many indicies we shift by
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

    // const auto w = tyvi::mdgrid_work{};

    std::array<runko::index_t,3> extents = extents_;

    auto kernel = [grid_mds = grid_->mds(), new_grid_mds = new_grid_->mds(), ax, min_rel, max_rel, t_min, t_max, order, extents, min_ind, max_ind] (const auto& idx) { // 
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

        new_grid_mds[min_inds][] += Interpolator(interpolation_values, t_min, order); 
        if (min_ind != max_ind) // Only add the second contribution if they are to different cells
            new_grid_mds[max_inds][] += Interpolator(interpolation_values, t_max, order);
    };

    w.for_each_index(*new_grid_, std::move(kernel)); // .wait()

    Clean(w); // Clean the old buffer and swap
}

void DenseGrid::TranslateZ(const tyvi::mdgrid_work& w, std::vector<VlasovGrid*> neighbors, value_type cfl){
    auto mds_grids = std::vector<decltype(grid_->mds())>();
    auto mds_new_grids = std::vector<decltype(new_grid_->mds())>();
    for (auto neighbor : neighbors){
        mds_new_grids.push_back(static_cast<DenseGrid*>(neighbor)->new_grid_->mds());
        mds_grids.push_back(static_cast<DenseGrid*>(neighbor)->grid_->mds());
    }

    const auto exs = extents_;
    const auto deltas = u_res_;

    auto kernel = [mds_grids, mds_new_grids, cfl, exs, deltas] (const auto& idx){
        const auto u = toolbox::Vec3(
            GetVelFromInd(idx[0], exs[0], deltas[0]),
            GetVelFromInd(idx[1], exs[1], deltas[1]),
            GetVelFromInd(idx[2], exs[2], deltas[2])
        );
        const value_type invGamma = value_type{1} / sstd::sqrt(value_type{1} + toolbox::dot(u,u));
        const value_type deltaZ = u[2]* invGamma * cfl;

        value_type min = sstd::floor(deltaZ); // the relative index of the cell with lower index we need to update
        value_type max = sstd::ceil(deltaZ); // the other relative index we need to update

        value_type t_min = min - deltaZ; 
        value_type t_max = max - deltaZ; 

        int32_t min_ind = static_cast<int32_t>(min) + static_cast<int32_t>((mds_grids.size()-1)/2);
        int32_t max_ind = static_cast<int32_t>(max) + static_cast<int32_t>((mds_grids.size()-1)/2);

        auto interpolation_values = std::vector<value_type>();
        interpolation_values.push_back(mds_grids[(mds_grids.size()-1)/2][idx][]); // TODO: add support for higher order interpolations, i.e. more values here

        mds_new_grids[min_ind][idx][] += Interpolator(interpolation_values, t_min, 0);
        if (min_ind != max_ind) // Only add the second contribution if they are to different cells
            mds_new_grids[max_ind][idx][] += Interpolator(interpolation_values, t_max, 0);
    };

    w.for_each_index(*grid_, std::move(kernel));
}

void DenseGrid::TranslateZ(std::vector<VlasovGrid*> neighbors, value_type cfl){
    const auto w = tyvi::mdgrid_work{};
    TranslateZ(w, neighbors, cfl);
    w.wait();
}

void DenseGrid::SendData(const tyvi::mdgrid_work& w, VlasovGrid &dest) const {
    try {
        DenseGrid &destination = dynamic_cast<DenseGrid&>(dest);

        // const auto w = tyvi::mdgrid_work{};

        auto kernel = [source_mds = new_grid_->mds(), dest_mds = destination.new_grid_->mds()] (const auto &idx) {
            dest_mds[idx][] += source_mds[idx][];
        };

        w.for_each_index(*grid_, std::move(kernel)); // .wait()

    } catch (const std::bad_cast& e) {
        throw std::runtime_error("Cannot send data to a VlasovGrid of a different type!\n");
    }
}

void DenseGrid::recv_data(const tyvi::mdgrid_work& w, const VlasovGrid &orig) {
    try {
        const DenseGrid &origin = dynamic_cast<const DenseGrid&>(orig);

        // const auto w = tyvi::mdgrid_work{};

        auto kernel = [dest_mds = grid_->mds(), source_mds = origin.grid_->mds()] (const auto &idx) {
            dest_mds[idx][] = source_mds[idx][];
        };

        w.for_each_index(*grid_, std::move(kernel)); // .wait()

    } catch (const std::bad_cast& e) {
        throw std::runtime_error("Cannot send data to a VlasovGrid of a different type!\n");
    }
}

void DenseGrid::Clean(const tyvi::mdgrid_work& w){
    w.for_each_index(*grid_, [grid_mds = grid_->mds()] (const auto& idx){ // Set grid to zeros in order to have a clean new_grid after the swap
        grid_mds[idx][] = static_cast<value_type>(0.0f);
    });

    std::swap(grid_, new_grid_);
}

void DenseGrid::Clean(){
    const auto w = tyvi::mdgrid_work{};
    Clean(w);
    w.wait();
}

vlv::VlasovGrid::value_type DenseGrid::DebugGetFluid(std::array<runko::index_t,3> inds) const{ // This function is inefficient; don't use for anything important
    const auto w = tyvi::mdgrid_work{};
    w.sync_to_staging(*grid_).sync_to_staging(*new_grid_).wait();
    return grid_->staging_mds()[inds][];
}


vlv::VlasovGrid::value_type DenseGrid::CalculateMoment(const tyvi::mdgrid_work& w, MomentCalculationFunction func) {
    namespace rn           = std::ranges;
    const auto grid_mds = grid_->mds(); 
    const auto index_space = tyvi::sstd::index_space(grid_mds);
    const auto exs = extents_;
    const auto deltas = u_res_;

    const auto calculate_integral = [grid_mds, exs, deltas, func](const auto idx) {
        const auto f = grid_mds[idx][];
        const auto u = toolbox::Vec3(
            GetVelFromInd(idx[0], exs[0], deltas[0]),
            GetVelFromInd(idx[1], exs[1], deltas[1]),
            GetVelFromInd(idx[2], exs[2], deltas[2])
        );
        const value_type gamma = sstd::sqrt(value_type{1} + toolbox::dot(u,u)); // TODO is it necessary to calculate gamma here?
        return f * static_cast<value_type>(func(u[0],u[1],u[2],gamma));
    };
    const auto v_iterator_begin =
        thrust::make_transform_iterator(index_space.begin(), calculate_integral);
    const auto v_iterator_end = rn::next(v_iterator_begin, rn::size(index_space));
    const auto du3 = deltas[0] * deltas[1] * deltas[2];
    return thrust::reduce(w.on_this(), v_iterator_begin, v_iterator_end) * du3;
}

constexpr vlv::VlasovGrid::value_type VlasovGrid::Interpolator(std::vector<value_type> &values, value_type t, runko::index_t order){

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
        msg << "Interpolator of order " << order << " is not implemented!\n";
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
