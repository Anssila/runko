import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib import animation
from matplotlib import image
import sys
import itertools

def read_snapshot(filename : str, species = 0):
    n_species = np.fromfile(filename, dtype=np.int64, count=1)[0]
    if species >= n_species:
        raise IndexError(f"Species {species} out of range of {n_species} species provided!")
    size = np.fromfile(filename, dtype=np.int64, count=1, offset=8)[0]
    extents = np.fromfile(filename, dtype=np.int64, count=6, offset=16)
    # print(f"Reading file {filename}, found {n_species} species with size {size} and extents {extents}!")
    data = np.fromfile(filename, dtype=np.float32, count=size, offset=64 + species * (size * 4 + 56)).reshape(extents)
    return data

if __name__ == "__main__":
    filepath = sys.argv[1]
    n_laps = 0
    while True:
        try:
            with open(f"{filepath}/vlv_snapshot(0,0,0)_{n_laps}.bin") as f:
                n_laps += 1
        except FileNotFoundError:
            break
    print(f"Found {n_laps} laps of data for filepath {filepath}")
    initial_data = read_snapshot(f"{filepath}/vlv_snapshot(0,0,0)_0.bin")
    shape = initial_data.shape
    print(f"Snapshot of shape {shape} found!")
    slice_x = shape[0]//2
    slice_ux = shape[3]//2
    slice = initial_data[slice_x,:,:,slice_ux,:,:]
    print(f"Showing middle slice with shape {slice.shape} and sliced indices {slice_x} & {slice_ux}!")
    fig, axs = plt.subplots(shape[1], shape[2])
    laps = sys.argv[2:]
    imgs = np.empty((shape[1], shape[2]), dtype=image.AxesImage)
    if laps != None and len(laps) == 1:
        lap = int(laps[0])
        if lap >= n_laps:
            print(f"Lap {lap} out of range of {n_laps} laps!")
        else:
            data = read_snapshot(f"{filepath}/vlv_snapshot(0,0,0)_{lap}.bin")[slice_x,:,:,slice_ux,:,:]
            for y, z in itertools.product(range(shape[1]), range(shape[2])):
                ax = axs[y,z]
                norm = colors.SymLogNorm(vmin=0, vmax=np.max(data), linthresh=1e-6)
                imgs[y,z] = ax.imshow(data[y,z,:,:], norm= norm, cmap="plasma")
            # fig.colorbar(imgs[0], ax =axs[0,0])
            plt.show()
    else:
        min = 0
        max = n_laps
        if (len(laps) == 2):
            min = int(laps[0])
            max = int(laps[1])
        print(f"Animating from frame {min} to {max}!")
        data = read_snapshot(f"{filepath}/vlv_snapshot(0,0,0)_0.bin")[slice_x,:,:,slice_ux,:,:]
        norm = colors.SymLogNorm(vmin=0, vmax=np.max(data), linthresh=1e-6)
        for y, z in itertools.product(range(shape[1]), range(shape[2])):
            ax = axs[y,z]
            imgs[y,z] = ax.imshow(data[y,z,:,:], norm= norm, cmap="plasma")

        def update(frame):
            data = read_snapshot(f"{filepath}/vlv_snapshot(0,0,0)_{frame + min}.bin")[slice_x,:,:,slice_ux,:,:]
            for y, z in itertools.product(range(shape[1]), range(shape[2])):
                imgs[y,z].set_data(data[y,z,:,:])
            return imgs

        ani = animation.FuncAnimation(fig, update, frames=max-min, interval=100, repeat_delay=500)
        plt.show()