import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

def read_snapshot(filename : str, species = 0):
    n_species = np.fromfile(filename, dtype=np.int64, count=1)[0]
    if species >= n_species:
        raise IndexError(f"Species {species} out of range of {n_species} species provided!")
    size = np.fromfile(filename, dtype=np.int64, count=1, offset=8)[0]
    extents = np.fromfile(filename, dtype=np.int64, count=6, offset=16)
    data = np.fromfile(filename, dtype=np.float32, count=size, offset=40 + species * (size * 4+32)).reshape(extents)
    return data

if __name__ == "__main__":
    n_tiles = 8
    fig, ax = plt.subplots()
    norm = colors.SymLogNorm(vmin=0, vmax=100, linthresh=1e-6)
    datas = []
    for i in range(n_tiles):
        data = read_snapshot(f"vlv_snapshot(0,0,{i}).bin")[1,1,:,1,1,:]
        datas.append(data)
    full_data = np.rot90(np.concatenate(datas, axis=0))
    img = ax.imshow(full_data, norm= norm)
    fig.colorbar(img, ax =ax)

    # img = ax.imshow()

    # fig.colorbar(imgs[0], ax =axs[i])

    # plt.show()