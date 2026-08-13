import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib import animation
import sys
import pickle

def read_snapshot(filename : str, species = 0):
    n_species = np.fromfile(filename, dtype=np.int64, count=1)[0]
    if species >= n_species:
        raise IndexError(f"Species {species} out of range of {n_species} species provided!")
    size = np.fromfile(filename, dtype=np.int64, count=1, offset=8)[0]
    extents = np.fromfile(filename, dtype=np.int64, count=6, offset=16)
    data = np.fromfile(filename, dtype=np.float32, count=size, offset=40 + species * (size * 4+32)).reshape(extents)
    return data

if __name__ == "__main__":
    n_tiles = 32
    fig, ax = plt.subplots()
    datas = []
    filepath = sys.argv[1]
    n_laps = 0


    ax.set_xlabel("Paikka")
    ax.set_ylabel("Itseisnopeus (c)")
    ax.set_title("1D1V faasiavaruuden lukumäärätiheys")
    config = None
    config_filename = f"{filepath}/config.pkl"
    try:
        with open(config_filename, 'rb') as file:
            config = pickle.load(file)
    except:
        print("Failed to open config file!")
    spatial_ex = config.n_tiles[2] * config.n_cells_per_tile[2]
    while True:
        try:
            with open(f"{filepath}/vlv_snapshot(0,0,0)_{n_laps}.bin") as f:
                n_laps += 1
        except FileNotFoundError:
            break
    print(f"Found {n_laps} laps of data for filepath {filepath}")
    laps = sys.argv[2:]
    min = 0
    max = n_laps
    if (len(laps)== 1):
        min = int(laps[0])
        max = int(laps[0])+1
        print(f"Showing frame {min}!")
    elif (len(laps) == 2):
        min = int(laps[0])
        max = int(laps[1])
        print(f"Animating from frame {min} to {max}!")
    else:
        print(f"Animating all frames from {min} to {max}!")
    for i in range(n_tiles):
        data = read_snapshot(f"{filepath}/vlv_snapshot(0,0,{i})_{min}.bin")[0,0,:,0,0,:]
        datas.append(data)
    full_data = np.rot90(np.concatenate(datas, axis=0))
    norm = colors.SymLogNorm(vmin=0, vmax=np.max(full_data), linthresh=1e-6)
    im = ax.imshow(full_data, norm= norm, aspect='auto', cmap="plasma", extent=[-spatial_ex//2,spatial_ex//2,-config.u_max[2],config.u_max[2]])

    def update(frame):
        datas = []
        for i in range(n_tiles):
            data = read_snapshot(f"{filepath}/vlv_snapshot(0,0,{i})_{frame + min}.bin")[0,0,:,0,0,:]
            datas.append(data)
        full_data = np.rot90(np.concatenate(datas, axis=0))
        im.set_data(full_data)

    ani = animation.FuncAnimation(fig, update, frames=max-min, interval=100)
    fig.colorbar(im, ax =ax, label="Lukumäärätiheys")
    plt.show()
    # ani.save(f"{filepath}/anim.gif", writer="pillow", dpi=400)
    ani.save(f"{filepath}/{min}.png", writer="pillow", dpi=400)