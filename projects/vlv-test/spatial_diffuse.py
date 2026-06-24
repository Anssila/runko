import runko
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation
from matplotlib.colors import LogNorm

def draw_slice(ax, slice : np.ndarray):
    ax.imshow(slice, cmap='viridis')
    # ax.colorbar()

def create_tile(x,y,z, spatial):
    config = runko.Configuration(None)
    config.Nx = 1
    config.Ny = 1
    config.Nz = 1
    config.NxMesh = 3
    config.NyMesh = 3
    config.NzMesh = spatial
    config.Nvx = x
    config.Nvy = y
    config.Nvz = z
    config.xmin = 0
    config.ymin = 0
    config.zmin = 0
    config.cfl = 0.5
    config.field_propagator = "FDTD2"
    # config.deltaUx = 0.06666
    # config.deltaUy = 0.2
    # config.deltaUz = 0.4
    config.inftyx = 0.1
    config.inftyy = 0.1
    config.inftyz = 2.0


    tile_grid_idx = (0,0,0)

    return runko.vlv.threeD.Tile(tile_grid_idx, config)

def maxwell_distr(vx, vy, vz):
    v_0 = 0.1 # refrence velocity = sqrt((2*k*T)/m) (m is mass, k is boltzmann const, T is temperature)
    return (np.pi*v_0**2)**(-1.5) * np.exp(-(vx**2+vy**2+vz**2)/v_0**2)

def plot_distr():
    plt.close()
    global distributions
    for d in distributions:
        plt.plot(d[0], d[1])
    plt.show()

def get_width(distr):
    max_v = max(distr)
    start = 0
    end = 0
    for i in range(len(distr)):
        if distr[i] > max_v/2 and start == 0:
            start = i
        if start != 0 and distr[i] < max_v/2:
            end = i
            break
    return end - start


def max_coords(data):
    max = 0.0
    maxx = 0
    maxy = 0
    maxz = 0
    for i in range(len(data)):
        for j in range(len(data[0])):
            for k in range(len(data[0][0])):
                if data[i][j][k] > max:
                    max = data[i][j][k]
                    maxx = i
                    maxy = j
                    maxz = k
    return (maxx, maxy, maxz)


def center(data):
    maxc = max_coords(data)
    distr = data[maxc[0]][maxc[1]]
    koords = np.array(list(range(len(distr)))) - maxc[2]

    return (koords, distr)
if __name__ == "__main__":

    exs = input("Set velocity space extents: ").split(" ")
    if len(exs) == 0 or exs[0] == "" or exs[0] == "0":
        exs = [100, 100, 100]
    dim = len(exs)

    if dim == 1:
        exs = [2,2,exs[0]]
    elif dim == 2:
        exs = [2,exs[0],exs[1]]

    spatial_ex = int(input("Set spatial extent: "))

    if spatial_ex < 3:
        spatial_ex = 3

    mode = input("Mode (anim/distr/width/temp): ")

    if mode != "anim" and mode != "distr" and mode != "width" and mode != "temp":
        raise RuntimeError("Invalid mode!")

    tot_iters = 50

    np.set_printoptions(linewidth=200)

    fig, ax = plt.subplots(1,spatial_ex+6)

    tile = create_tile(int(exs[0]),int(exs[1]),int(exs[2]), spatial_ex)

    v_init = lambda x,y,z : maxwell_distr(x if dim == 3 else 0,y if dim >= 2 else 0,z)
    v_0 = lambda x,y,z : 0
    middle = spatial_ex // 2
    tile.SetVelDistribution(0,0,middle,v_init)
    for i in range(-3, spatial_ex +3):
        if i != middle:
            tile.SetVelDistribution(0,0,i,v_0)
    tot = 0
    for i in range(spatial_ex):
        grid = tile.GetVelDistribution(0,0,i)
        tot += sum(sum(sum(grid)))
    # distributions = [center(grid)]
    itercounts = [0]

    print(f"Total fluid: {tot}")

    iters = 0
    cbar = None
    im = []

    if mode == "anim":
        grid = tile.GetVelDistribution(0,0,i)

        if dim == 3:
            for i in range(-3, spatial_ex +3):
                im.append(ax[i+3].imshow(sum(grid), norm=LogNorm(vmin=1e-8, vmax=maxwell_distr(0,0,0))))
            cbar = fig.colorbar(im[-1], ax=ax[-1])
        elif dim == 2:
            for i in range(-3, spatial_ex +3):
                im.append(ax[i+3].imshow(grid[0], norm=LogNorm(vmin=1e-8, vmax=maxwell_distr(0,0,0))))
            cbar = fig.colorbar(im[-1], ax=ax[-1])
        else:
            im.append(ax[i+3].plot(list(range(len(grid[0][0]))),grid[0][0])[0])
        ims = []
    def update(frame):
        global grid, iters

        if frame == 0:
            tile.SetVelDistribution(0,0,middle,v_init)
            for i in range(-3, spatial_ex +3):
                if i != middle:
                    tile.SetVelDistribution(0,0,i,v_0)
            iters += 1
        extra_iters = 1
        for i in range(extra_iters):
            tile.Translate()
            tile.DebugBC()
            tile.CleanUp()
            # x_acc = 0 if dim < 3 else -np.sin(-(frame*extra_iters+i)/1) * 0.5
            # y_acc = 0 if dim < 2 else -np.cos(-(frame*extra_iters+i)/1) * 0.5
            # z_acc =  np.sin(-(frame*extra_iters+i)/1) * 0.5 
            # tile.DebugAccelerate(0,0,0,x_acc, y_acc, z_acc, 1.0)
            iters += 1

        if frame % 1 == 0:
            # grid = tile.GetVelDistribution(0,0,0)
            # distributions.append(center(grid))
            # itercounts.append(iters)
            tot2 = 0
            for i in range(spatial_ex):
                grid = tile.GetVelDistribution(0,0,i)
                tot2 += sum(sum(sum(grid)))
            print(f"Total fluid: {tot2}, difference {(tot2/tot-1)*100} % of original")

        # print(f"Total fluid: {sum(sum(grid[0]))}")
        if mode == "anim":
            for i in range(-3, spatial_ex +3):
                grid = tile.GetVelDistribution(0,0,i)
                if dim == 3:
                    im[i+3].set_array(sum(grid))#[max_coords(grid)[0]]
                    # cbar[i+3].update_normal(im[i])
                    im[i+3].set_clim(vmin=1e-8, vmax=maxwell_distr(0,0,0))
                elif dim == 2:
                    im[i+3].set_array(grid[0])#[max_coords(grid)[0]]
                    # cbar[i+3].update_normal(im[i])
                    im[i+3].set_clim(vmin=1e-8, vmax=maxwell_distr(0,0,0))
                else:
                    im[i+3].set_ydata(grid[0][0])    

            # im.set_clim(vmin=grid[0].min(), vmax=grid[0].max())


            return im

    if mode == "anim":
        ani = animation.FuncAnimation(fig, update, frames=tot_iters, interval=50, blit=True, repeat_delay=1000)
        plt.show()
    # elif mode == "distr":
    #     for i in range(tot_iters):
    #         update(i)

    #     plot_distr()
    # elif mode == "width":
    #     for i in range(tot_iters):
    #         update(i)
    #     widths = []
    #     for d in distributions:
    #         widths.append(get_width(d[1]))

    #     k,b = np.polyfit(itercounts[5:],widths[5:], 1)

    #     print(f"Slope: {k} cells / iteration")

    #     plt.plot(itercounts,widths, "o")
    #     plt.plot([0,itercounts[-1]], [b,k*itercounts[-1]+b], "-")
    #     plt.show() 
    # elif mode == "temp":
    #     for i in range(tot_iters):
    #         update(i)
    #     widths = []
    #     for d in distributions:
    #         widths.append(get_width(d[1]))

    #     temps = np.sqrt(widths)
    #     plt.plot(itercounts, temps, "o")
    #     plt.show()

