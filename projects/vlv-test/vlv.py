import runko
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation
from matplotlib.colors import LogNorm

def draw_slice(ax, slice : np.ndarray):
    ax.imshow(slice, cmap='viridis')
    # ax.colorbar()
    
def create_tile(x,y,z):
    config = runko.Configuration(None)
    config.Nx = 1
    config.Ny = 1
    config.Nz = 1
    config.NxMesh = 5
    config.NyMesh = 5
    config.NzMesh = 5
    config.Nvx = x
    config.Nvy = y
    config.Nvz = z
    config.xmin = 0
    config.ymin = 0
    config.zmin = 0
    config.cfl = 1
    config.field_propagator = "FDTD2"
    # config.deltaUx = 0.06666
    # config.deltaUy = 0.2
    # config.deltaUz = 0.4
    config.inftyx = 5.0
    # config.inftyy = 2.0
    # config.inftyz = 3.0
    
    
    tile_grid_idx = (0,0,0)
    
    return runko.vlv.threeD.Tile(tile_grid_idx, config)

def maxwell_distr(vx, vy, vz):
    v_0 = 1.1 # refrence velocity = sqrt((2*k*T)/m) (m is mass, k is boltzmann const, T is temperature)
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
    
    exs = input("Set extents: ").split(" ")
    if len(exs) == 0 or exs[0] == "" or exs[0] == "0":
        exs = [100, 100, 100]
    dim = len(exs)
    
    if dim == 1:
        exs = [1,1,exs[0]]
    elif dim == 2:
        exs = [1,exs[0],exs[1]]
    
    mode = input("Mode (anim/distr/width/temp): ")
    
    if mode != "anim" and mode != "distr" and mode != "width" and mode != "temp":
        raise RuntimeError("Invalid mode!")
    
    tot_iters = 1000
    
    np.set_printoptions(linewidth=200)
    
    fig, ax = plt.subplots()
    
    tile = create_tile(int(exs[0]),int(exs[1]),int(exs[2]))

    v_init = lambda x,y,z : maxwell_distr(x if dim == 3 else 0,y if dim >= 2 else 0,z)
    tile.SetVelDistribution(0,0,0,v_init)
    grid = tile.GetVelDistribution(0,0,0)
    tot = sum(sum(sum(grid)))
    distributions = [center(grid)]
    itercounts = [0]
    
    print(f"Total fluid: {tot}")
    
    iters = 0
    cbar = None
    im = None
    
    if mode == "anim":
        if dim > 1:
            im = ax.imshow(grid[0], norm=LogNorm(vmin=1e-8, vmax=maxwell_distr(0,0,0))) 
            cbar = fig.colorbar(im, ax=ax)
        else:
            im = ax.plot(list(range(len(grid[0][0]))),grid[0][0])[0]
        ims = []
    
    def update(frame):
        global grid, iters
        
        if frame == 0:
            tile.SetVelDistribution(0,0,0,v_init)
            tile.DebugAccelerate(0,0,0,0.0,0.0,0,1.0)
            iters += 1
        extra_iters = 5
        for i in range(extra_iters):
            x_acc = 0 if dim < 3 else -np.sin(-(frame*extra_iters+i)/1) * 0.5
            y_acc = 0 if dim < 2 else -np.cos(-(frame*extra_iters+i)/1) * 0.5
            z_acc =  np.sin(-(frame*extra_iters+i)/1) * 0.5 
            tile.DebugAccelerate(0,0,0,x_acc, y_acc, z_acc, 1.0)
            iters += 1
        
        if frame % 10 == 0:
            grid = tile.GetVelDistribution(0,0,0)
            distributions.append(center(grid))
            itercounts.append(iters)
            print(f"Total fluid: {sum(sum(sum(grid)))}, difference {(1-sum(sum(sum(grid)))/tot)*100} % of original")
        
        # print(f"Total fluid: {sum(sum(grid[0]))}")
        if mode == "anim":
            grid = tile.GetVelDistribution(0,0,0)
            
            if dim > 1:
                im.set_array(grid[max_coords(grid)[0]])
                cbar.update_normal(im)
                im.set_clim(vmin=1e-8, vmax=maxwell_distr(0,0,0))
            else:
                im.set_ydata(grid[0][0])    
            
            # im.set_clim(vmin=grid[0].min(), vmax=grid[0].max())
            

            return [im]

    if mode == "anim":
        ani = animation.FuncAnimation(fig, update, frames=tot_iters, interval=50, blit=True, repeat_delay=1000)
        plt.show()
    elif mode == "distr":
        for i in range(tot_iters):
            update(i)
        
        plot_distr()
    elif mode == "width":
        for i in range(tot_iters):
            update(i)
        widths = []
        for d in distributions:
            widths.append(get_width(d[1]))
            
        k,b = np.polyfit(itercounts[5:],widths[5:], 1)
        
        print(f"Slope: {k} cells / iteration")
        
        plt.plot(itercounts,widths, "o")
        plt.plot([0,itercounts[-1]], [b,k*itercounts[-1]+b], "-")
        plt.show() 
    elif mode == "temp":
        for i in range(tot_iters):
            update(i)
        widths = []
        for d in distributions:
            widths.append(get_width(d[1]))
        
        temps = np.sqrt(widths)
        plt.plot(itercounts, temps, "o")
        plt.show()
