#pragma once

#include "../mdgrid_common.h"

namespace vlv{

// Base class for different kinds of vlasov grid implementations
class VlasovGrid{
public:
    using value_type = float;

protected:
    value_type deltaV_; // The spacing / resolution of the velocity space discretization

public: 
    // implement shift operator using strang-splitting
    void Shift(value_type dx, value_type dy, value_type dz, value_type dt){
        // TODO do correct strang-splitting, for now just do full shift sequentially for every dir
        Shift_dir(dx*dt, 0);
        Shift_dir(dy*dt, 1);
        Shift_dir(dz*dt, 2);
    } 
    virtual void InitZero() = 0;
    virtual value_type GetTotalFluid() const = 0;

private: 
    // Shifts in the coordinate axes are private and virtual because they are used by the strang splitting
    // main shift function that is public (these shouldn't be directly accessed) and they are implemented in the 
    // actual implementations of the vlasov grid (dense grid / sparse grid ...)
    virtual void Shift_dir(value_type dv, size_t ax = 0) = 0; // Shift the velocity space values in the ax-direction by dv
};


// Simple densely stored grid implementation of a vlasov grid
class DenseGrid : public VlasovGrid{
public:
    using VelGrid = runko::ScalarGrid<value_type>;

private:
    std::array<std::size_t, 3> extents_;
    std::array<value_type, 3> infty_; // The max value for u that can be stored in the dense grid, for each axis
    std::array<value_type, 3> deltaU_; // The resolution for u, i.e. what is the difference in u of neighboring cells of the dense grid, for each axis
    VelGrid *grid_, *new_grid_; // the actual grids that store the velocity space phase fluid 
    // new_grid_ is used to update values and the pointers are swapped every time

    value_type dummy;

public:
    DenseGrid(std::size_t Nx, std::size_t Ny, std::size_t Nz);
    ~DenseGrid(){
        delete grid_;
        delete new_grid_;
    }
    value_type GetDummy() const {return dummy;}
    value_type GetTotalFluid() const override;
    void InitZero() override;
    void SetInfty(std::array<value_type,3> inftys); // Set the max value in the dense grid (infty_), sets deltaU accordingly based on extents
    void SetDelta(std::array<value_type,3> deltas); // Set the resolution of the dense grid (deltaU), sets infty accordingly based on extents
    size_t GetIndexFromVel(value_type u, size_t ax = 0) const; // Helper function to get the index in the sparse grid corresponding to a velocity in the ax-direction
    value_type GetVelFromIndex(size_t ind, size_t ax = 0) const; // Helper function to get the velocity (beta in the ax-direction) corresponding to an index in the sparse grid

private:
    void Shift_dir(value_type dv, size_t ax = 0) override; // Function for shifting in 1D along ax    
};

} // namespace vlv