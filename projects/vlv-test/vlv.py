import runko
import numpy as np
import matplotlib.pyplot as plt


def maxwell_distr(vx, vy, vz, v_0):# reference velocity = sqrt((2*k*T)/m) (m is mass, k is boltzmann const, T is temperature)
    return (np.pi*v_0**2)**(-0.5) * np.exp(-(vx**2+vy**2+vz**2)/v_0**2)


if __name__ == "__main__":
    config = runko.Configuration(None)
    config.io_outdir = "full_6D_test"
    config.tile_partitioning = "hilbert_curve"
    config.n_laps = 100
    config.n_tiles = [1, 1, 1]
    config.n_cells_per_tile = [7, 7, 7]
    config.v_grid_extents = [21,21,21]
    config.u_max = [1.0,1.0,1.0]
    config.cfl = 1.0

    config.skin_depth = 1.0

    config.omega_p = config.cfl / config.skin_depth
    config.field_propagator = "fdtd2"
    config.m0 = 1.0
    config.n0 = 1.0
    config.q0 = config.omega_p * np.sqrt(config.m0/config.n0) # q = sqrt(omega_p^2*m/n)
    v_T = config.u_max[2]/10.0
    # v_0 = config.u_max[2]/4.0
    noise_A = 1e-1
    noise_f = 2.0*np.pi/(config.n_cells_per_tile[2]*config.n_tiles[2])

    E0 = lambda x, y, z: (np.sin(noise_f*x) * noise_A, np.sin(noise_f*y) * noise_A, np.sin(noise_f*z) * noise_A)
    B0 = lambda x, y, z: (0, 0, 0)
    J0 = lambda x, y, z: (0, 0, 0)

    tile_grid = runko.TileGrid(config)

    actual_n = config.n0
    n_0 = config.n0


    logger = runko.runko_logger()

    def vlv0(x,y,z,ux,uy,uz):
        global v_T #, v_0, actual_n, n_0
        return (np.pi*v_T**2)**(-1.5) * np.exp(-(uz**2+uy**2+ux**2)/v_T**2)
        #n_0/actual_n * (np.pi*v_T**2)**(-0.5) * (np.exp(-(uz-v_0)**2/v_T**2) + np.exp(-(uz+v_0)**2/v_T**2)) * (1+np.cos(noise_f*z) % noise_A)


    for idx in tile_grid.local_tile_indices():
        tile = runko.vlv.threeD.Tile(idx, config)
        tile.set_EBJ(E0, B0, J0)
        tile.set_vlv(vlv0, 0)
        tile_grid.add_tile(tile, idx)

    # Simulation config:
    simulation = tile_grid.configure_simulation(config)

    def lap_function(x):
        # if simulation.lap % 50 == 0:
        x.io_emf_snapshot()
        x.grid_write_vlv_snapshot()
        # if simulation.lap % 50 == 0:
        x.io_average_E_energy_density()

        x.grid_accelerate()
        x.grid_deposit_current()

        x.comm_external(runko.tools.comm_mode.vlv_particle)
        x.comm_local(runko.tools.comm_mode.vlv_particle)

        x.grid_translate()

        # if simulation.lap == 100:

        x.grid_add_current()

        if simulation.lap % 10 == 0:
            simulation.log_timer_statistics()

    simulation.for_each_lap(lap_function)
    simulation.log_timer_statistics()