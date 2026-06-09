#include "tile.h"

namespace vlv {

template<std::size_t D>
Tile<D>::Tile(
    std::array<std::size_t, 3> tile_grid_indices, 
    const toolbox::ConfigParser& conf) :
    corgi::Tile<D>(),
    emf::Tile<D>(tile_grid_indices, conf), 
    vel_grid_(
        static_cast<runko::index_t>(conf.get_or_throw<std::size_t>("Nvx")),
        static_cast<runko::index_t>(conf.get_or_throw<std::size_t>("Nvy")),
        static_cast<runko::index_t>(conf.get_or_throw<std::size_t>("Nvz"))
    )
    {
    auto deltaUx = conf.get<float>("deltaUx");
    auto deltaUy = conf.get<float>("deltaUy");
    auto deltaUz = conf.get<float>("deltaUz");
    auto x_infty = conf.get<float>("inftyx");
    auto y_infty = conf.get<float>("inftyy");
    auto z_infty = conf.get<float>("inftyz");

    if      (deltaUx.has_value() && !x_infty.has_value()) vel_grid_.SetDelta({deltaUx.value(), deltaUy.value_or(deltaUx.value()), deltaUz.value_or(deltaUx.value())}); 
    else if (!deltaUx.has_value() && x_infty.has_value()) vel_grid_.SetInfty({x_infty.value(), y_infty.value_or(x_infty.value()), z_infty.value_or(x_infty.value())}); 
    else {
        std::stringstream msg;
        if (!deltaUx.has_value() && !x_infty.has_value())
            msg << "Cannot create tile without either deltaU or infty!\n";
        else
            msg << "Cannot create tile with both deltaU and infty!\n";
        throw std::runtime_error(msg.str());
    }

    vel_grid_.InitZero();
}

template<std::size_t D>
void Tile<D>::DebugAccelerate(double ax, double ay, double az, double dt){
    auto ax_ = static_cast<value_type>(ax);
    auto ay_ = static_cast<value_type>(ay);
    auto az_ = static_cast<value_type>(az);
    auto dt_ = static_cast<value_type>(dt);
    vel_grid_.Shift(ax_, ay_, az_, dt_);
}

template<std::size_t D>
void Tile<D>::SetVelGrid(VDF distribution){
    vel_grid_.SetGridData(distribution);
}

} // namespace vlv

template class vlv::Tile<3>;
