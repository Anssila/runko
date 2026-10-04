import numpy as np
import sys, os
import pickle
import matplotlib
import matplotlib.pyplot as plt
from runko.mpiio_reader import read_field_snapshot
from matplotlib import animation

def read_snapshot(filename : str, species = 0):
    n_species = np.fromfile(filename, dtype=np.int64, count=1)[0]
    if species >= n_species:
        raise IndexError(f"Species {species} out of range of {n_species} species provided!")
    size = np.fromfile(filename, dtype=np.int64, count=1, offset=8)[0]
    extents = np.fromfile(filename, dtype=np.int64, count=6, offset=16)
    data = np.fromfile(filename, dtype=np.float32, count=size, offset=64 + species * (size * 4+32)).reshape(extents)
    return data

def read_full_box(path, var_name):
    return read_field_snapshot(path)[var_name]

#-------------------------------------------------- 
if __name__ == "__main__":

    fig = plt.figure(1, figsize=(6.0, 3.0)) # single-column figure
    #fig = plt.figure(1, figsize=(7.0,  2.5)) # two-column figure

    # add ticks to both sides 
    plt.rc('xtick', top   = True)
    plt.rc('ytick', right = True)

    plt.rc('font',  family='serif',)
    plt.rc('text',  usetex=False)

    # make labels slightly smaller 
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    plt.rc('axes',  labelsize=8)
    plt.rc('legend',  handlelength=4.0)

    # number of rows and columns for the figure
    nrow_fig = 2
    ncol_fig = 1

    gs = plt.GridSpec(nrow_fig, ncol_fig, height_ratios=[2, 1])
    gs.update(wspace = 0.25)
    gs.update(hspace = 0.3)

    axs = np.empty( (nrow_fig,ncol_fig), dtype=object)

    # for j in range(ncol_fig):
    #     for i in range(nrow_fig):
    #         axs[i,j] = plt.subplot(gs[i,j])
    #         axs[i,j].minorticks_on()

    axs[0,0] = plt.subplot(gs[0,0])
    axs[1,0] = plt.subplot(gs[1,0], sharex=axs[0,0])

    axs[0,0].minorticks_on()
    axs[1,0].minorticks_on()

    # axs[0,0].set_xlabel(r"Paikka, $x~(d_\mathrm{s})$")
    axs[0,0].set_ylabel(r"Itseisnopeus, $u~(c)$")

    # axs[0,0].set_xlim((0, 5))
    # axs[0,0].set_ylim((1e-1, 1e2))

    #axs[0,0].set_xscale('log')
    # axs[0,0].set_yscale('log')


    axs[1,0].set_xlabel(r"Paikka, $x~(d_\mathrm{s})$")
    axs[1,0].set_ylabel(r"$E~(\mathrm{statV}/d_\mathrm{s})$")

    # axs[1,0].set_xlim((0, 5))
    # axs[1,0].set_ylim((1e-1, 1e2))


    # optional colorbar
    tmin = 0.0
    tmax = 10.0
    # norm = matplotlib.colors.Normalize(vmin=tmin, vmax=tmax)
    cmap = matplotlib.colormaps['plasma']


    #--------------------------------------------------
    # figure 
    filepath = sys.argv[1]
    frames = sys.argv[2:]
    config = None
    config_filename = f"{filepath}/config.pkl"
    try:
        with open(config_filename, 'rb') as file:
            config = pickle.load(file)
    except:
        print("Failed to open config file!")
    spatial_ex = config.n_tiles[2] * config.n_cells_per_tile[2]
    n_tiles = config.n_tiles[2]

    datas = []

    for i in range(n_tiles):
        data = read_snapshot(f"{filepath}/vlv_snapshot(0,0,{i})_{frames[0]}.bin")[0,0,:,0,0,:]
        datas.append(data)

    full_data = np.rot90(np.concatenate(datas, axis=0))
    norm = matplotlib.colors.SymLogNorm(vmin=0, vmax=1.0, linthresh=1e-4)
    x_ex = spatial_ex / config.skin_depth
    im = axs[0,0].imshow(full_data, norm= norm, aspect='auto', cmap="plasma", extent=[-x_ex/2, x_ex/2,-config.u_max[2],config.u_max[2]])


    E_data = read_full_box(f"{filepath}/flds_{frames[1]}.bin", "ez")[:,1,1]
    E_data *= 3.4526e+2 * config.skin_depth
    x_data = np.linspace(-x_ex/2, x_ex/2, len(E_data))

    axs[1,0].plot(x_data, E_data, color = "C0", alpha = 1.0, lw = 1.0, linestyle="solid")
    
    # def update(frame):
    #     datas = []
    #     for i in range(n_tiles):
    #         data = read_snapshot(f"{filepath}/vlv_snapshot(0,0,{i})_{frame + min}.bin")[0,0,:,0,0,:]
    #         datas.append(data)
    #     full_data = np.rot90(np.concatenate(datas, axis=0))
    #     im.set_data(full_data)

    # for i, a in enumerate([3,4,5,6]):
    #     col = 'C' + str(i)

    #     xx = np.linspace(0, 10, 100)
    #     yy = xx**a

    #     axs[0,0].plot(xx, yy, 
    #                   color=col,
    #                   alpha = 1.0,
    #                   lw = 1.0,
    #                   #drawstyle='steps-pre',
    #                   linestyle='solid',
    #                   )



    #--------------------------------------------------
    # save 

    # control these (in units of [0,1]) to position the figure
    axleft    = 0.18
    axbottom  = 0.16
    axright   = 0.96
    axtop     = 0.80

    #--------------------------------------------------
    if True: # optional colorbar
        pos1 = axs[0,0].get_position()
        axwidth  = axright - axleft
        axheight = (axtop - axbottom)*0.03
        axpad = 0.02
        cax = fig.add_axes([axleft, axtop + axpad, axwidth, axheight])

        cb1 = matplotlib.colorbar.ColorbarBase(
                cax,
                cmap=cmap,
                norm=norm,
                orientation='horizontal',
                ticklocation='top')
        cb1.set_label(r'Faasiavaruuden lukumäärätiheys $f~(n_0)$')
    #--------------------------------------------------

    fig.subplots_adjust(left=axleft, bottom=axbottom, right=axright, top=axtop)


    # ani = animation.FuncAnimation(fig, update, frames=list(range(max-min))[::], interval=200)

    # plt.show()

    fname = f'{filepath}/presentation_{frames[0]}.pdf' 
    plt.savefig(fname, dpi=300)

    # fname = 'fig.png' 
    # plt.savefig(fname, dpi=300)