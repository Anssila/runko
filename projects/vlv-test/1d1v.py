import runko
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation
from matplotlib.colors import LogNorm

def create_tile(x,y,z, spatial, v_max):
    config = runko.Configuration(None)
    config.n_tiles = [1,1,1]
    config.n_cells_per_tile = [3,3,spatial]
    config.v_grid_extents = [x,y,z]
    config.xmin = 0
    config.ymin = 0
    config.zmin = 0
    config.cfl = 50.5
    config.field_propagator = "fdtd2"
    config.u_max = [0.1,0.1,v_max]

    tile_grid_idx = (0,0,0)


    tile_grid_idx = (0,0,0)

    return runko.vlv.threeD.Tile(tile_grid_idx, config)

def maxwell_distr(vx, vy, vz):
    v_0 = 2.1 # refrence velocity = sqrt((2*k*T)/m) (m is mass, k is boltzmann const, T is temperature)
    return (np.pi*v_0**2)**(-0.5) * np.exp(-(vx**2+vy**2+vz**2)/v_0**2)

if __name__ == "__main__":

    vel_ex = int(input("Set velocity space extent: "))
    spatial_ex = int(input("Set spatial extent: "))
    filename = input("Save animation? Filename (leave blank to show and not save): ")

    v_max = 0.01

    if spatial_ex < 3:
        spatial_ex = 3

    tot_iters = 50

    np.set_printoptions(linewidth=200)

    fig, ax = plt.subplots()

    tile = create_tile(3,3,vel_ex, spatial_ex, v_max)

    v_init = lambda x,y,z : 0.1 #maxwell_distr(0,0,z)
    v_0 = lambda x,y,z : 0
    middle = spatial_ex // 2
    tile.SetVelDistribution(1,1,middle,v_init)
    for i in range(-3, spatial_ex +3):
        if i != middle:
            tile.SetVelDistribution(1,1,i,v_0)
    tot = 0
    for i in range(spatial_ex):
        grid = tile.GetVelDistribution(1,1,i)
        tot += sum(sum(sum(grid)))
    # distributions = [center(grid)]
    itercounts = [0]

    print(f"Total fluid: {tot}")

    iters = 0
    data = []
    for i in range(spatial_ex):
        grid = tile.GetVelDistribution(1,1,i)
        data.append(grid[0][0])
    data = np.array(data)
    data = np.rot90(data)
    cmap = plt.get_cmap('plasma').copy()

    cmap.set_bad(color='#100788')#

    masked_data = np.ma.masked_less(data, 1e-8)

    im = ax.imshow(masked_data, norm=LogNorm(vmin=1e-8, vmax=maxwell_distr(0,0,0)), cmap=cmap, interpolation='none', extent=[-middle,spatial_ex-middle,-v_max,v_max])
    ax.set_aspect(vel_ex/(2*v_max))
    ax.set_xlabel("Paikka")
    ax.set_ylabel("Itseisnopeus (c)")
    ax.set_title("1D1V faasiavaruuden lukumäärätiheys")
    cbar = fig.colorbar(im, ax=ax, label="Lukumäärätiheys")


    def update(frame):
        global grid, iters, data

        if frame == 0:
            tile.SetVelDistribution(1,1,middle,v_init)
            for i in range(-3, spatial_ex +3):
                if i != middle:
                    tile.SetVelDistribution(1,1,i,v_0)
            iters += 1
        extra_iters = 10
        for i in range(extra_iters):
            tile.Translate()
            tile.DebugBC()
            tile.CleanUp()
            iters += 1

        if frame % 1 == 0:
            tot2 = 0
            for i in range(spatial_ex):
                grid = tile.GetVelDistribution(1,1,i)
                tot2 += sum(sum(sum(grid)))
            print(f"Total fluid: {tot2}, difference {(tot2/tot-1)*100} % of original")

        for i in range(spatial_ex):
            grid = tile.GetVelDistribution(1,1,i)
            data = np.rot90(data, 3)


            for i in range(spatial_ex):
                grid = tile.GetVelDistribution(1,1,i)
                data[i] = grid[0][0]
            data = np.rot90(data)
            masked_data = np.ma.masked_less(data, 1e-8)

            im.set_array(masked_data)
            # im.set_clim(vmin=1e-7, vmax=0.02) #maxwell_distr(0,0,0)

        return [im]

    ani = animation.FuncAnimation(fig, update, frames=tot_iters, interval=100, blit=True, repeat_delay=1000)
    if filename == "":
        plt.show()
    else:
        ani.save(filename=filename, writer="pillow")