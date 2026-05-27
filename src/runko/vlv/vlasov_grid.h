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
    // TODO: should the shift amount be float / value_type / double??
    void Shift(value_type dx, value_type dy, value_type dz){
        // TODO do correct strang-splitting
        Shift_x(dx);
        Shift_y(dy);
        Shift_z(dz);
    } 
    virtual void InitZero() = 0;
    virtual value_type GetTotalFluid() = 0;

private: 
    // Shifts in the coordinate axes are private and virtual because they are used by the strang splitting
    // main shift function that is public (these shouldn't be directly accessed) and they are implemented in the 
    // actual implementations of the vlasov grid (dense grid / sparse grid ...)
    virtual void Shift_x(value_type dv) = 0; // Shift the velocity space values in the x-direction by dv
    virtual void Shift_y(value_type dv) = 0; // Shift the velocity space values in the y-direction by dv
    virtual void Shift_z(value_type dv) = 0; // Shift the velocity space values in the z-direction by dv
};


// Simple densely stored grid implementation of a vlasov grid
class DenseGrid : public VlasovGrid{
public:
    using VelGrid = runko::ScalarGrid<value_type>;

private:
    std::array<std::size_t, 3> extents_;
    VelGrid grid_;
    
    value_type dummy;

public:
    DenseGrid(std::size_t Nx, std::size_t Ny, std::size_t Nz);
    value_type GetDummy() {return dummy;}
    value_type GetTotalFluid() override;
    void InitZero() override;

private:
    void Shift_x(value_type dv) override;
    void Shift_y(value_type dv) override;
    void Shift_z(value_type dv) override;
};

} // namespace vlv