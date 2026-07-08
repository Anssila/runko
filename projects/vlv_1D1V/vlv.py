import runko
import numpy as np
import matplotlib.pyplot as plt


def maxwell_distr(vx, vy, vz, v_0):# reference velocity = sqrt((2*k*T)/m) (m is mass, k is boltzmann const, T is temperature)
    return (np.pi*v_0**2)**(-0.5) * np.exp(-(vx**2+vy**2+vz**2)/v_0**2)


if __name__ == "__main__":
    config = runko.Configuration(None)

    config.tile_partitioning = "hilbert_curve"
    config.n_laps = 100
    config.n_tiles = [1, 1, 1]
    config.n_cells_per_tile = [5, 5, 20]
    config.v_grid_extents = [3,3,20]
    config.u_max = [2.0,2.0,2.0]
    config.cfl = 0.5
    config.field_propagator = "fdtd2"
    config.q0 = 1.0
    config.m0 = 1.0
    v_0 = config.u_max[2]/20.0

    tile_grid = runko.TileGrid(config)

    # Initial conditions:

    k = 2 * np.pi * np.array((0, 0, 1)) / 10.

    def E0(x, y, z):
        r = np.array((x, y, z))
        return np.sin(np.dot(k, r)), 0, 0

    B0 = lambda x, y, z: (0, 0, 0)
    J0 = lambda x, y, z: (0, 0, 0)

    def vlv0(x,y,z,ux,uy,uz):
        if x != 3 or y != 3 or ux != 0.0 or uy != 0.0:
            return 0.0
        global v_0
        return (np.pi*v_0**2)**(-0.5) * (np.exp(-(uz-10.0*v_0)**2/v_0**2) + np.exp(-(uz+10.0*v_0)**2/v_0**2))

    for idx in tile_grid.local_tile_indices():
        tile = runko.vlv.threeD.Tile(idx, config)
        tile.set_EBJ(E0, B0, J0)
        tile.set_vlv()
        tile_grid.add_tile(tile, idx)

    # Simulation config:
    simulation = tile_grid.configure_simulation(config)

    def lap_function(x):
        x.grid_Translate()
        x.grid_DebugBC()
        x.grid_CleanUp()
        x.grid_accelerate()
        x.grid_deposit_current()
        x.grid_add_current()
        if simulation.lap % 5 == 0:
            x.io_emf_snapshot()

        if simulation.lap % 5 == 0:
            simulation.log_timer_statistics()

    simulation.for_each_lap(lap_function)
