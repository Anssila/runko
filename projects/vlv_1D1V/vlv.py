import runko
import numpy as np
import matplotlib.pyplot as plt
import random

def maxwell_distr(vx, vy, vz, v_0):# reference velocity = sqrt((2*k*T)/m) (m is mass, k is boltzmann const, T is temperature)
    return (np.pi*v_0**2)**(-0.5) * np.exp(-(vx**2+vy**2+vz**2)/v_0**2)


if __name__ == "__main__":
    config = runko.Configuration(None)
    config.io_outdir = "two-stream"
    config.tile_partitioning = "hilbert_curve"
    config.n_tiles = [1, 1, 32]
    config.n_cells_per_tile = [3, 3, 32]
    config.v_grid_extents = [3,3,512]
    config.u_max = [18.0,18.0,18.0]
    config.cfl = 0.90

    config.skin_depth = 28.810122 / 3.0

    config.n0 = 1.0
    config.m0 = 1.0
    config.io_outdir = "simulations/long_sim"
    #"simulations/rel_test_C_" + str(round(config.cfl*100)) + "_R_" + str(round(config.skin_depth)) + "_" + str(config.v_grid_extents[2]) + "X" + str(config.n_cells_per_tile[2]*config.n_tiles[2]) #config.io_outdir + "_c_" + str(round(config.cfl)) + "_R_" + str(round(config.skin_depth))

    config.omega_p = config.cfl / config.skin_depth

    config.n_laps = 500.0 / config.omega_p

    config.field_propagator = "fdtd2"
    config.q0 = config.omega_p * np.sqrt(config.m0/config.n0) # q = sqrt(omega_p^2*m/n)
    v_T = config.u_max[2]/config.v_grid_extents[2]*10.0
    v_0 = np.sqrt(8.0)#config.u_max[2]/6.0
    config.v_T = v_T
    config.v_0 = v_0
    config.gamma_b = np.sqrt(1 + v_0**2)
    config.gamma_m = config.omega_p * config.gamma_b**-1.5 # calculate maximum growth rate
    k_m = np.sqrt(3)/2*config.omega_p / v_0 / config.cfl * config.gamma_b**-0.5# calculate wave number for the maximally growing mode

    config.max_wavelength = 2*np.pi/k_m
    config.n_cycles = k_m * (config.n_cells_per_tile[2] * config.n_tiles[2]) / (2*np.pi)

    noise_A = 1e-7
    # noise_f =0.5*np.pi/(config.n_cells_per_tile[2]*config.n_tiles[2])
    noise_f = np.round(config.n_cycles) * 2.0*np.pi/(config.n_cells_per_tile[2]*config.n_tiles[2])

    def rand(seed):
        random.seed(seed)
        return random.random()

    E0 = lambda x, y, z: (0, 0, rand(z)*noise_A - 0.5 * noise_A + np.sin(noise_f*z) * (noise_A*20.0))#np.sin(noise_f*z) % noise_A - 0.5 * noise_A)##np.sin(noise_f*z)*noise_A)
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
        return n_0/actual_n * (np.pi*v_T**2)**(-0.5) * (np.exp(-(uz-v_0)**2/v_T**2) + np.exp(-(uz+v_0)**2/v_T**2)) #* (1+np.cos(noise_f*z) % noise_A)

    test_tile = runko.vlv.threeD.Tile((0,0,0), config)
    test_tile.set_vlv(vlv0,0)
    actual_n = 0.5 * test_tile.CalculateMoment(0,0,0, lambda x, y, z, gamma : 1.0, 0)
    print(f"Actual n: {actual_n}, fixing!")

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

        if simulation.lap % 50 == 0:
            x.grid_write_vlv_snapshot()

        x.grid_add_current()

        # x.comm_external(runko.tools.comm_mode.emf_E)
        # x.comm_local(runko.tools.comm_mode.emf_E)
        # x.grid_push_e()


        if simulation.lap % 10 == 0:
            simulation.log_timer_statistics()

    simulation.for_each_lap(lap_function)
    simulation.log_timer_statistics()