#pragma once

#include "../mdgrid_common.h"

namespace vlv{

// Base class for different kinds of vlasov grid implementations
class VlasovGrid{
public:
    using value_type = float;
    using VelocityDistributionFunction = std::function<double(double, double, double)>; // Function to set the VDF, inputs are velocity coordinates in x, y, and z directions and output is the amount of phase fluid
    using MomentCalculationFunction = std::function<double(double, double, double, double)>; // Function to calculate a moment of the velocity space, inputs are velocity coordinates (x,y,z) as well as a pre-calculated gamma

protected:    
    // Values (2*order + 1) must be centered around the point relative to which t is measured, returns interpolation result to given order
    static constexpr value_type Interpolator(std::vector<value_type> &values, value_type t, const runko::index_t order=0); 
public: 


    // implement shift operator using strang-splitting
    void Shift(const tyvi::mdgrid_work& w, [[maybe_unused]] value_type dx, [[maybe_unused]] value_type dy, value_type dz){
        // TODO do correct strang-splitting, for now just do full shift sequentially for every dir 
        // TODO or should shift be done fully 3d?
        // Shift_dir(w, dx, 0);
        // Shift_dir(w, dy, 1);
        Shift_dir(w, dz, 2);
    }
    // overload Shift for a non-async version 
    void Shift(value_type dx, value_type dy, value_type dz){ 
        const auto w = tyvi::mdgrid_work{};
        Shift(w, dx, dy, dz);
        w.wait();
    }
    virtual void InitZero() = 0; // Function to initialize the velocity distribution to zeros
    virtual void InitDelta(std::array<value_type,3> v) = 0;  // Function to initialize the velocity distribution to a delta function around the specified velocity v
    virtual value_type DebugGetTotalFluid() const = 0;
    virtual void SetGridData(VelocityDistributionFunction distribution) = 0;
    virtual void set_size(runko::index_t Nx, runko::index_t Ny, runko::index_t Nz) = 0;
    virtual void set_u_max(std::array<value_type,3> max) = 0; // Set the max value in the dense grid (u_max_), sets u_res_ accordingly based on extents
    virtual void set_u_res(std::array<value_type,3> res) = 0; // Set the resolution of the dense grid (u_res_), sets u_max_ accordingly based on extents
    virtual void TranslateZ(const tyvi::mdgrid_work& w, std::vector<VlasovGrid*> neighbors, value_type cfl) = 0; // Translate the fluid in the z-axis from this VlasovGrid to neighboring grids depending on the velocity space coordinates
    virtual void TranslateZ(std::vector<VlasovGrid*> neighbors, value_type cfl) = 0; // A non-async overload of Translate
    virtual void Clean(const tyvi::mdgrid_work& w) = 0; // Clean the old buffer and swap
    virtual void Clean() = 0; // A non-async overload of Clean
    virtual void SendData(const tyvi::mdgrid_work& w, VlasovGrid &dest) = 0; // Send the data of this (virtual) VlasovGrid to another VlasovGrid such that it is superimposed on the data of the destination grid (the values are summed into the new grid)

    virtual value_type CalculateMoment(const tyvi::mdgrid_work& w, MomentCalculationFunction func) = 0; // Generalized function for calculating moments of the distribution velocity distribution.

    private: 
    // Shifts in the coordinate axes are private and virtual because they are used by the strang splitting
    // main shift function that is public (these shouldn't be directly accessed) and they are implemented in the 
    // actual implementations of the vlasov grid (dense grid / sparse grid ...)
    virtual void Shift_dir(const tyvi::mdgrid_work& w, value_type dv, runko::index_t ax, const runko::index_t order = 0) = 0; // Shift the velocity space values in the ax-direction by dv
};


// Simple densely stored grid implementation of a vlasov grid
class DenseGrid : public VlasovGrid{
public:
    using VelGrid = runko::ScalarGrid<value_type>;

private:
    std::array<runko::index_t, 3> extents_;
    std::array<value_type, 3> u_max_; // The max value for u that can be stored in the dense grid, for each axis
    std::array<value_type, 3> u_res_; // The resolution for u, i.e. what is the difference in u of neighboring cells of the dense grid, for each axis
    std::shared_ptr<VelGrid> grid_, new_grid_; // the actual grids that store the velocity space phase fluid 
    // new_grid_ is used to update values and the pointers are swapped every time

public:
    DenseGrid() = default;
    DenseGrid(runko::index_t Nx, runko::index_t Ny, runko::index_t Nz) { set_size(Nx,Ny,Nz); }

    void set_size(runko::index_t Nx, runko::index_t Ny, runko::index_t Nz) override;

    void InitZero() override;
    void InitDelta(std::array<value_type,3>) override;
    void set_u_max(std::array<value_type,3> max) override;
    void set_u_res(std::array<value_type,3> res) override;

    void SetGridData(VelocityDistributionFunction distribution) override;

    void TranslateZ(const tyvi::mdgrid_work& w, std::vector<VlasovGrid*> neighbors, value_type cfl) override;
    void TranslateZ(std::vector<VlasovGrid*> neighbors, value_type cfl) override;
    void Clean(const tyvi::mdgrid_work& w) override;
    void Clean() override;
    void SendData(const tyvi::mdgrid_work& w, VlasovGrid &test) override;

    value_type CalculateMoment(const tyvi::mdgrid_work& w, MomentCalculationFunction func) override;

    std::array<runko::index_t,3> GetIndFromVel(std::array<value_type,3> u) const; // Helper function to get the indicies corresponding to a velocity in the sparse grid
    std::array<value_type,3> GetVelFromInd(std::array<runko::index_t,3> inds) const; // Helper function to get the velocity corresponding to a set of indicies in the dense grid

    value_type DebugGetTotalFluid() const override;
    value_type DebugGetFluid(std::array<runko::index_t,3> inds) const; // Debug function to get the fluid in a grid cell
    auto GetStagingMDS() const { 
        const auto w = tyvi::mdgrid_work{};
        w.sync_to_staging(*grid_).wait();
        return grid_->staging_mds(); 
    }
    auto GetMDS() const { return grid_->mds(); }
    auto GetExtents() const { return extents_; }

private:
    void Shift_dir(const tyvi::mdgrid_work& w, value_type dv, runko::index_t ax, const runko::index_t order=0) override; // Function for shifting in 1D along ax, interpolated to order "order"
    inline static void ClampInds(std::array<int32_t,3> &inds, std::array<runko::index_t,3> ex);
    static constexpr runko::index_t GetIndFromVel(value_type u, runko::index_t ex, value_type deltaU); // Helper function to get the index in the dense grid corresponding to a velocity in the ax-direction
    static constexpr value_type GetVelFromInd(runko::index_t ind, runko::index_t ex, value_type deltaU); // Helper function to get the velocity (beta in the ax-direction) corresponding to an index in the dense grid
};

// Concept for templating Tiles (and VlasovContainers) based on the implementation of VlasovGrid that it uses
template <typename VGrid>
concept VelGridType = std::derived_from<VGrid, VlasovGrid>;

} // namespace vlv