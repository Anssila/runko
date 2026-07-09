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

    tot_iters = 1000
    config = runko.Configuration(None)

    config.tile_partitioning = "hilbert_curve"
    config.n_laps = tot_iters
    config.n_tiles = [1, 1, 1]
    config.n_cells_per_tile = [3, 3, spatial_ex]
    config.v_grid_extents = [3,3,vel_ex]
    config.u_max = [2.0,2.0,2.0]
    config.cfl = 0.5
    config.field_propagator = "fdtd2"
    config.q0 = 0.02
    config.q1 = 0.02
    config.m0 = 1.0
    config.m1 = 1.0
    v_0 = config.u_max[2]/40.0

    noise_A = 1e-7
    noise_f = 4*np.pi/spatial_ex

    E0 = lambda x, y, z: (0, 0, np.sin(noise_f*z)*noise_A)
    B0 = lambda x, y, z: (0, 0, 0)
    J0 = lambda x, y, z: (0, 0, 0)

    def vlv0(x,y,z,ux,uy,uz):
        # if abs(x-0.5) > 0.01 or abs(y-0.5) > 0.01 or abs(ux) > 1e-6 or abs(uy) > 1e-6:
        #     return 0
        global v_0
        return (np.pi*v_0**2)**(-0.5) * (np.exp(-(uz-10.0*v_0)**2/v_0**2) ) * (1+np.cos(noise_f*z)*noise_A)

    def vlv1(x,y,z,ux,uy,uz):
        # if abs(x-1.5) > 0.01 or abs(y-1.5) > 0.01 or abs(ux) > 1e-6 or abs(uy) > 1e-6:
        #     return 0
        global v_0
        return (np.pi*v_0**2)**(-0.5) * (np.exp(-(uz+10.0*v_0)**2/v_0**2)) * (1+np.cos(noise_f*z)*noise_A)

    tile = runko.vlv.threeD.Tile((0,0,0), config)
    tile.set_EBJ(E0, B0, J0)
    tile.set_vlv(vlv0, 0)
    tile.set_vlv(vlv1, 1)

    data0 = np.rot90(tile.get_vlv_snapshot(0)[0,0,:,1,1,:])
    data1 = np.rot90(tile.get_vlv_snapshot(1)[0,0,:,1,1,:])
    (E0x, E0y, E0z), (B0x, B0y, B0z), (J0x, J0y, J0z) = tile.get_EBJ()
    e_data = E0z[1][1]
    j_data = J0z[1][1]
    np.set_printoptions(linewidth=200,precision=4,threshold=3*3*spatial_ex*3*3*vel_ex)
    # print(data)
    fig, axs = plt.subplots(1,2)
    im = []
    im.append(axs[0].imshow(data0 + data1, norm=LogNorm(vmin=1e-8, vmax=(np.pi*v_0**2)**(-0.5)), extent=[-spatial_ex//2,spatial_ex//2,-config.u_max[2],config.u_max[2]]))
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
    axs[1].set_ylabel("Numeerinen arvo ($\\hat{E}$, $\\hat{J}$)")
    axs[1].set_title("Sähkökenttä ja virrantiheys")

    def lap_function(frame):
        if frame == 0:
            tile.set_EBJ(E0, B0, J0)
            tile.set_vlv(vlv0, 0)
            tile.set_vlv(vlv1, 1)
            data0 = np.rot90(tile.get_vlv_snapshot(0)[0,0,:,1,1,:])
            data1 = np.rot90(tile.get_vlv_snapshot(1)[0,0,:,1,1,:])
            im[0].set_array(data0 + data1)
            (E0x, E0y, E0z), (B0x, B0y, B0z), (J0x, J0y, J0z) = tile.get_EBJ()
            e_data = E0z[1][1]
            j_data = J0z[1][1]
            im[1].set_ydata(e_data)
            axs[1].set_ylim(min(min(e_data),min(j_data)),max(max(e_data),max(j_data)))
            im[2].set_ydata(j_data)
            return im
        for i in range(5):
            tile.Translate()
            tile.DebugBC()
            tile.CleanUp()
            tile.accelerate()
            tile.deposit_current()
            tile.add_current()
        data0 = np.rot90(tile.get_vlv_snapshot(0)[0,0,:,1,1,:])
        data1 = np.rot90(tile.get_vlv_snapshot(1)[0,0,:,1,1,:])
        im[0].set_array(data0 + data1)
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
        print(f"Total energy in E field: {tile.get_tot_energy_E()}")
        return im


    ani = animation.FuncAnimation(fig, lap_function, frames=tot_iters, interval=75, blit=False, repeat_delay=1000)
    plt.show()
    if filename != "":
        ani.save(filename=filename, writer="pillow")
