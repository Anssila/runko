# Copyright 2026 - 2026, Miro Palmu, Joonas Nättilä and the runko contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from runko.mpiio_reader import read_field_snapshot


# read simulation output file and reshape to python format
def read_full_box(path, var_name):
    return read_field_snapshot(path)[var_name]

def read_je(path_to_h5: str):
    """
    FIXME: this might be broken after removing hdf5
    """
    jx = read_full_box(path_to_h5, "jx")
    jy = read_full_box(path_to_h5, "jy")
    jz = read_full_box(path_to_h5, "jz")
    ex = read_full_box(path_to_h5, "ex")
    ey = read_full_box(path_to_h5, "ey")
    ez = read_full_box(path_to_h5, "ez")

    return jx * ex + jy * ey + jz * ez


def plot(data, fig, ax, vmin=None, vmax=None, cblabel=""):

    zx_data = np.rot90(data[:, 1, :])
    vmin = vmin if vmin else np.min(zx_data)
    vmax = vmax if vmax else np.max(zx_data)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    img = ax.imshow(zx_data, norm=norm)
    ax.set_title("")

    fig.colorbar(img,
                 ax=ax,
                 orientation=None,
                 label=cblabel)

if __name__ == "__main__":

    var = sys.argv[1]
    filenames = sys.argv[2:] 
    filenames.sort(key = lambda name : int(name[name.find("flds")+5:-4]))
    rows = len(filenames)
    fig, ax_all = plt.subplots(rows, 1,
                               layout="compressed",
                               figsize=(10, rows))

    if rows == 1:
        ax_all = [ax_all]

    for i, file in enumerate(filenames):
        ax = ax_all[i]

        match var:
            case "je":
                plot(read_je(file), fig, ax, cblabel=file)
            case "bz":
                plot(read_full_box(file, "bz"), fig, ax, cblabel=file)
            case "ez":
                plot(read_full_box(file, "ez"), fig, ax, cblabel=file)
            case "jz":
                plot(read_full_box(file, "jz"), fig, ax, cblabel=file)


    fig.suptitle(var)
    plt.show()
