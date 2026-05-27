#include "vlasov_grid.h"

namespace vlv{

DenseGrid::DenseGrid(std::size_t Nx, std::size_t Ny, std::size_t Nz){
    extents_ = {Nx, Ny, Nz};
    grid_ = VelGrid(Nx, Ny, Nz);
    if (Nx != 5) throw std::runtime_error { "Vlasov grid initialized wrong!\n" }; // To test that the test actually works
};

void DenseGrid::Shift_x(float dv){
    dummy += dv;
}

void DenseGrid::Shift_y(float dv){
    dummy += dv;
}

void DenseGrid::Shift_z(float dv){
    dummy += dv;
}

} // namespace vlv