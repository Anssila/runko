import runko
import numpy as np
import matplotlib.pyplot as plt


def maxwell_distr(vx, vy, vz, v_0):# reference velocity = sqrt((2*k*T)/m) (m is mass, k is boltzmann const, T is temperature)
    return (np.pi*v_0**2)**(-0.5) * np.exp(-(vx**2+vy**2+vz**2)/v_0**2)


if __name__ == "__main__":
    config = runko.Configuration(None)
    config.io_outdir = "two-stream"
    config.tile_partitioning = "hilbert_curve"
    config.n_laps = 1000
    config.n_tiles = [1, 1, 8]
    config.n_cells_per_tile = [3, 3, 40]
    config.v_grid_extents = [3,3,360]
    config.u_max = [1.0,1.0,1.0]
    config.cfl = 1.0

    config.skin_depth = 40.0

    config.omega_p = config.cfl / config.skin_depth
    config.field_propagator = "fdtd2"
    config.m0 = 1.0
    config.n0 = 1.0
    config.q0 = config.omega_p * np.sqrt(config.m0/config.n0) # q = sqrt(omega_p^2*m/n)
    v_T = config.u_max[2]/100.0
    v_0 = config.u_max[2]/4.0
    noise_A = 1e-6
    noise_f = 0.5*np.pi/(config.n_cells_per_tile[2]*config.n_tiles[2])

    E0 = lambda x, y, z: (0, 0, np.sin(noise_f*z) % noise_A - 0.5 * noise_A)
    B0 = lambda x, y, z: (0, 0, 0)
    J0 = lambda x, y, z: (0, 0, 0)

    tile_grid = runko.TileGrid(config)

    actual_n = config.n0
    n_0 = config.n0


    logger = runko.runko_logger()

    def vlv0(x,y,z,ux,uy,uz):
        # if abs(x-0.5) > 0.01 or abs(y-0.5) > 0.01 or abs(ux) > 1e-6 or abs(uy) > 1e-6:
        #     return 0
        global v_T, v_0, actual_n, n_0
        return n_0/actual_n * (np.pi*v_T**2)**(-0.5) * (np.exp(-(uz-v_0)**2/v_T**2) + np.exp(-(uz+v_0)**2/v_T**2)) * (1+np.cos(noise_f*z) % noise_A)


    for idx in tile_grid.local_tile_indices():
        tile = runko.vlv.threeD.Tile(idx, config)
        tile.set_EBJ(E0, B0, J0)
        tile.set_vlv(vlv0, 0)
        tile_grid.add_tile(tile, idx)

    # Simulation config:
    simulation = tile_grid.configure_simulation(config)

    # def sync_E(x):
    #     x.comm_external(runko.tools.comm_mode.emf_E)
    #     x.comm_local(runko.tools.comm_mode.emf_E)

    # simulation.prelude(sync_E)

    def lap_function(x):
        if simulation.lap % 50 == 0:
            x.io_emf_snapshot()
        if simulation.lap % 50 == 0:
            x.io_average_E_energy_density()

        x.grid_accelerate()
        x.grid_deposit_current()

        x.comm_external(runko.tools.comm_mode.vlv_particle)
        x.comm_local(runko.tools.comm_mode.vlv_particle)

        x.grid_Translate()
        x.grid_CleanUp()

        # if simulation.lap == 100:
        #     x.grid_write_vlv_snapshot()

        x.grid_add_current()

        # x.comm_external(runko.tools.comm_mode.emf_E)
        # x.comm_local(runko.tools.comm_mode.emf_E)
        # x.grid_push_e()


        if simulation.lap % 10 == 0:
            simulation.log_timer_statistics()

    simulation.for_each_lap(lap_function)
    simulation.log_timer_statistics()