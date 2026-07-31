import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

def read_snapshot(filename : str, species = 0):
    n_species = np.fromfile(filename, dtype=np.int64, count=1)[0]
    if species >= n_species:
        raise IndexError(f"Species {species} out of range of {n_species} species provided!")
    size = np.fromfile(filename, dtype=np.int64, count=1, offset=8)[0]
    extents = np.fromfile(filename, dtype=np.int64, count=3, offset=16)
    data = np.fromfile(filename, dtype=np.float32, count=size, offset=40 + species * (size * 4+32)).reshape(extents)
    return data

if __name__ == "__main__":
    n_tiles = 8
    fig, axs = plt.subplots(1, n_tiles)
    if n_tiles == 1:
        axs = [axs]
    i = 0
    imgs = []
    norm = colors.SymLogNorm(vmin=0, vmax=100.0, linthresh=1e-3)
    for i in range(n_tiles):
        data = read_snapshot(f"vlv_snapshot(0,0,{i}).bin")
        imgs.append(axs[i].imshow(data[0], norm= norm))
    fig.colorbar(imgs[0], ax =axs[i])

    plt.show()