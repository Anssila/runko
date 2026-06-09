#pragma once

#include "../mdgrid_common.h"

namespace vlv{


// Base class for different kinds of vlasov grid implementations
class VlasovGrid{
public:
    using value_type = float;
    using VelocityDistributionFunction = std::function<double(double, double, double)>;

protected:
    value_type deltaV_; // The spacing / resolution of the velocity space discretization
public: 
    // implement shift operator using strang-splitting
    void Shift(value_type dx, value_type dy, value_type dz, value_type dt){
        // TODO do correct strang-splitting, for now just do full shift sequentially for every dir 
        // TODO or should shift be done fully 3d?
        Shift_dir(dx*dt, 0);
        Shift_dir(dy*dt, 1);
        Shift_dir(dz*dt, 2);
    } 
    virtual void InitZero() = 0; // Function to initialize the velocity distribution to zeros
    virtual void InitDelta(std::array<value_type,3> v) = 0;  // Function to initialize the velocity distribution to a delta function around the specified velocity v
    virtual value_type DebugGetTotalFluid() const = 0;
    virtual void SetGridData(VelocityDistributionFunction distribution) = 0;

private: 
    // Shifts in the coordinate axes are private and virtual because they are used by the strang splitting
    // main shift function that is public (these shouldn't be directly accessed) and they are implemented in the 
    // actual implementations of the vlasov grid (dense grid / sparse grid ...)
    virtual void Shift_dir(value_type dv, runko::index_t ax, const runko::index_t order = 0) = 0; // Shift the velocity space values in the ax-direction by dv
};


// Simple densely stored grid implementation of a vlasov grid
class DenseGrid : public VlasovGrid{
public:
    using VelGrid = runko::ScalarGrid<value_type>;

private:
    std::array<runko::index_t, 3> extents_;
    std::array<value_type, 3> infty_; // The max value for u that can be stored in the dense grid, for each axis
    std::array<value_type, 3> deltaU_; // The resolution for u, i.e. what is the difference in u of neighboring cells of the dense grid, for each axis
    std::unique_ptr<VelGrid> grid_, new_grid_; // the actual grids that store the velocity space phase fluid 
    // new_grid_ is used to update values and the pointers are swapped every time

public:
    DenseGrid(runko::index_t Nx, runko::index_t Ny, runko::index_t Nz);

    void InitZero() override;
    void InitDelta(std::array<value_type,3>) override;
    void SetInfty(std::array<value_type,3> inftys); // Set the max value in the dense grid (infty_), sets deltaU accordingly based on extents
    void SetDelta(std::array<value_type,3> deltas); // Set the resolution of the dense grid (deltaU), sets infty accordingly based on extents

    void SetGridData(VelocityDistributionFunction distribution) override;

    std::array<runko::index_t,3> GetIndFromVel(std::array<value_type,3> u) const; // Helper function to get the indicies corresponding to a velocity in the sparse grid
    std::array<value_type,3> GetVelFromInd(std::array<runko::index_t,3> inds) const; // Helper function to get the velocity corresponding to a set of indicies in the dense grid

    value_type DebugGetTotalFluid() const override;
    value_type DebugGetFluid(std::array<runko::index_t,3> inds) const; // Debug function to get the fluid in a grid cell
    auto GetStagingMDS() const { return grid_->staging_mds(); }
    auto GetMDS() const { return grid_->mds(); }
    auto GetExtents() const { return extents_; }
    // auto GetUnderlyingBuffer const { return grid_->span(); }
    std::vector<value_type> DebugGetGrid() const;

private:
    void Shift_dir(value_type dv, runko::index_t ax, const runko::index_t order=0) override; // Function for shifting in 1D along ax, interpolated to order "order"
    
    // Values (2*order + 1) must be centered around the point relative to which t is measured, returns interpolation result to given order
    static constexpr value_type Interpolator(std::vector<value_type> &values, value_type t, const runko::index_t order=0); 
    inline static void ClampInds(std::array<int32_t,3> &inds, std::array<runko::index_t,3> ex);
    runko::index_t GetIndFromVel(value_type u, runko::index_t ax) const; // Helper function to get the index in the dense grid corresponding to a velocity in the ax-direction
    value_type GetVelFromInd(runko::index_t ind, runko::index_t ax) const; // Helper function to get the velocity (beta in the ax-direction) corresponding to an index in the dense grid
};

} // namespace vlv