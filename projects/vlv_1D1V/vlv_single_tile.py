import runko
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm

def maxwell_distr(vx, vy, vz, v_0):# reference velocity = sqrt((2*k*T)/m) (m is mass, k is boltzmann const, T is temperature)
    return (np.pi*v_0**2)**(-0.5) * np.exp(-(vx**2+vy**2+vz**2)/v_0**2)


if __name__ == "__main__":

    vel_ex = max(int(input("Set velocity space extent: ")),3)
    spatial_ex = max(int(input("Set spatial extent: ")),3)
    filename = input("Save animation? Filename (leave blank to show and not save): ")

    totE = []
    analytic_y = []

    tot_iters = 1000
    extra_loops = 5
    config = runko.Configuration(None)

    config.tile_partitioning = "hilbert_curve"
    config.n_laps = tot_iters
    config.n_tiles = [1, 1, 1]
    config.n_cells_per_tile = [3, 3, spatial_ex]
    config.v_grid_extents = [3,3,vel_ex]
    config.u_max = [1.0,1.0,1.0]
    config.cfl = 0.5

    skin_depth = 30.0

    omega_p = config.cfl / skin_depth

    config.field_propagator = "fdtd2"
    config.m0 = 1.0
    config.n0 = 1.0
    config.q0 = omega_p * np.sqrt(config.m0/config.n0) # q = sqrt(omega_p^2*m/n)
    v_T = config.u_max[2]/100.0
    v_0 = config.u_max[2]/4.0

    noise_A = 1e-6
    noise_f = 0.5*np.pi/spatial_ex

    E0 = lambda x, y, z: (0, 0, np.sin(noise_f*z) % noise_A - 0.5 * noise_A)
    B0 = lambda x, y, z: (0, 0, 0)
    J0 = lambda x, y, z: (0, 0, 0)

    actual_n = config.n0
    n_0 = config.n0

    def vlv0(x,y,z,ux,uy,uz):
        # if abs(x-0.5) > 0.01 or abs(y-0.5) > 0.01 or abs(ux) > 1e-6 or abs(uy) > 1e-6:
        #     return 0
        global v_T, v_0, actual_n, n_0
        return n_0/actual_n * (np.pi*v_T**2)**(-0.5) * (np.exp(-(uz-v_0)**2/v_T**2) + np.exp(-(uz+v_0)**2/v_T**2)) * (1+np.cos(noise_f*z) % noise_A)

    tile = runko.vlv.threeD.Tile((0,0,0), config)
    tile.set_EBJ(E0, B0, J0)
    tile.set_vlv(vlv0, 0)
    tile.DebugBC()
 
    # average number density of plasma in the simulation
    actual_n = 0.5 * sum([tile.CalculateMoment(0,0,i, lambda x, y, z, gamma : 1.0, 0) for i in range(spatial_ex)])/spatial_ex # TODO: use both or just one species here?

    tile.set_vlv(vlv0, 0)
    tile.DebugBC()

    # omega_p = np.sqrt(config.cfl * config.q0**2/config.m0*avg_n) # calculate plasma frequency using avg_n
    gamma_m = omega_p # calculate maximum growth rate
    k_m = np.sqrt(3)/4*omega_p / v_0 / config.cfl# calculate wave number for the maximally growing mode (/4 because we have symmetrical beams)

    energy_0 = 0.0

    print(f"Plasma freq: {omega_p}")
    print(f"Maximum growth rate: {gamma_m}")
    print(f"Wave length of maximum growing mode: {2*np.pi/k_m}")
    print(f"Estimated number of cycles: {k_m * spatial_ex / (2*np.pi)}")

    data0 = np.rot90(tile.get_vlv_snapshot(0)[0,0,:,1,1,:])
    (E0x, E0y, E0z), (B0x, B0y, B0z), (J0x, J0y, J0z) = tile.get_EBJ()
    e_data = E0z[1][1]
    j_data = J0z[1][1]
    np.set_printoptions(linewidth=200,precision=4,threshold=3*3*spatial_ex*3*3*vel_ex)
    # print(data)
    fig, axs = plt.subplots(1,2)
    im = []
    im.append(axs[0].imshow(data0, norm=LogNorm(vmin=1e-8, vmax=(np.pi*v_T**2)**(-0.5)), extent=[-spatial_ex//2,spatial_ex//2,-config.u_max[2],config.u_max[2]]))
    cbar = fig.colorbar(im[0], ax=axs[0], label="Lukumäärätiheys")
    im.append(axs[1].plot(range(-spatial_ex//2,spatial_ex//2),e_data,label="Ez")[0])
    im.append(axs[1].plot(range(-spatial_ex//2,spatial_ex//2),j_data,label="Jz")[0])
    axs[1].legend()
    mins = [min(min(e_data),min(j_data))*1.5]
    maxs = [max(max(e_data),max(j_data))*1.5]
    axs[0].set_aspect(vel_ex/(2*config.u_max[2]))
    axs[0].set_xlabel("Paikka")
    axs[0].set_ylabel("Itseisnopeus (c)")
    axs[0].set_title("1D1V faasiavaruuden lukumäärätiheys")
    axs[1].set_xlabel("Paikka")
    axs[1].set_ylabel("Numeerinen arvo $\\hat{E}$, $\\hat{J}$")
    axs[1].set_title("Sähkökenttä ja virrantiheys")

    def lap_function(frame):
        global totE, analytic_y, energy_0
        if frame == tot_iters-1:
            plt.close()
        if frame == 0:
            tile.set_EBJ(E0, B0, J0)
            tile.set_vlv(vlv0, 0)
            tile.DebugBC()
            data0 = np.rot90(tile.get_vlv_snapshot(0)[0,0,:,1,1,:])
            im[0].set_array(data0)
            (E0x, E0y, E0z), (B0x, B0y, B0z), (J0x, J0y, J0z) = tile.get_EBJ()
            e_data = E0z[1][1]
            j_data = J0z[1][1]
            im[1].set_ydata(e_data)
            axs[1].set_ylim(min(min(e_data),min(j_data)),max(max(e_data),max(j_data)))
            im[2].set_ydata(j_data)
            energy_0 = tile.get_tot_energy_E()
            totE = [(energy_0, 0)]
            analytic_y = [energy_0]
            return im
        for i in range(extra_loops):
            tile.accelerate()
            tile.deposit_current()
            tile.add_current()
            tile.Translate()
            tile.CleanUp()
            tile.DebugBC()

        data0 = np.rot90(tile.get_vlv_snapshot(0)[0,0,:,1,1,:])
        im[0].set_array(data0)
        (E0x, E0y, E0z), (B0x, B0y, B0z), (J0x, J0y, J0z) = tile.get_EBJ()
        e_data = E0z[1][1]
        j_data = J0z[1][1]
        mins.append(min(min(e_data),min(j_data))*1.5)
        maxs.append(max(max(e_data),max(j_data))*1.5)
        if len(mins) > 10:
            mins.pop(0)
            maxs.pop(0)
        axs[1].set_ylim(sum(mins)/len(mins),sum(maxs)/len(maxs))
        im[1].set_ydata(e_data)
        im[2].set_ydata(j_data)
        energy = tile.get_tot_energy_E()
        # print(f"Total energy in E field: {energy}")
        totE.append((energy,frame*extra_loops*omega_p))
        analytic_y.append(energy_0 * np.exp(gamma_m*frame*extra_loops))
        return im


    ani = animation.FuncAnimation(fig, lap_function, frames=tot_iters, interval=75, blit=False, repeat_delay=1000)
    plt.show()
    if filename != "":
        ani.save(filename=filename, writer="pillow")

    y_data, x_data = zip(*totE)
    plt.plot(x_data, y_data, label="simulation")
    plt.plot(x_data, analytic_y, label="theory")
    plt.yscale("log")
    plt.xlabel("Aika ($\\omega_p^{-1}$)")
    plt.ylabel("Sähkökentän energia $\\langle \\hat{E}^2 \\rangle / 8\\pi$")
    plt.show()
