# Copyright 2026 - 2026, Miro Palmu, Joonas Nättilä and the runko contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from runko.mpiio_reader import read_field_snapshot
import pickle
import runko

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

def plot_energy(datas, fig, ax, names, config : runko.Configuration):
    i = 0
    for data in datas:
        x_data, y_data = zip(*data)
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        ax.semilogy(x_data * config.omega_p, y_data, label=names[i])
        analytic_y = 0.02*y_data[0] * np.exp(x_data * config.omega_p * 1.5)
        ax.semilogy(x_data * config.omega_p, analytic_y, label=names[i] + "_analytic")

        i += 1
    ax.set_title("Sähkökentän keskimääräinen energiatiheys aika-askeleen funktiona")
    ax.set_xlabel("Aika-askel")
    ax.set_ylabel("Sähkökentän energia $\\langle \\hat{E}^2 \\rangle / 8\\pi$")



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

def plot_in_time(datas, fig, ax):
    all_data = []
    for data in datas:
        z_data = data[:,1,1]
        vmin = min(z_data)
        vmax = max(z_data)
        z_data = (z_data - 0.5 * (vmax + vmin)) / (0.5 * (vmax-vmin))
        all_data.append(z_data)
    all_data = np.array(all_data)
    print(f"Min: {vmin}, Max: {vmax}")
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)#, linthresh=1e-7
    img = ax.imshow(all_data, norm=norm)#, cmap='seismic'

    fig.colorbar(img,
                 ax=ax,
                 orientation=None)

def plot_energy(datas, fig, ax : plt.Axes, names, configs : list[runko.Configuration]):
    for i in range(len(datas)):
        config = configs[i]
        data = datas[i]
        print(f"Parametrit simulaatiolle {names[i]}:")
        print(f"Plasmataajuus: {config.omega_p}")
        gamma_b = np.sqrt(1 + config.v_0**2)
        gamma_m = config.omega_p * gamma_b**-1.5 # calculate maximum growth rate
        print(f"Maksimaalinen kasvunopeus: {gamma_m}")
        print(f"cfl: {config.cfl}")
        print(f"skin_depth: {config.skin_depth}")
        print()
        x_data, y_data = zip(*data)
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        # ax.semilogy(x_data * config.omega_p, y_data, label=names[i])
        mults = [0.00008, 0.0002]
        analytic_y = y_data[0] * np.exp(x_data * gamma_m)

        log_data = np.log(y_data)
        log_analytic = np.log(analytic_y)

        index = -1
        for j in range(len(x_data)):
            if x_data[j] >= 100.0/config.omega_p:
                index = j
                break

        log_data -= log_data[index]
        log_analytic -= log_analytic[index]
        fmt = "-"
        if i > 3*len(datas)//4:
            fmt = ":"
        elif i > len(datas)//2:
            fmt = "-."
        elif i > len(datas)//4:
            fmt = "--"
        ax.plot(x_data * config.omega_p, log_data, fmt, label=names[i])
        if i == 0:
            ax.plot(x_data * config.omega_p, log_analytic, label=names[i] + "_analytic")
        # ax.semilogy(x_data * config.omega_p, analytic_y, label=names[i] + "_analytic")

    ax.set_title("Sähkökentän keskimääräinen energiatiheys aika-askeleen funktiona")
    ax.set_xlabel("Aika-askel")
    ax.set_ylabel("Sähkökentän energia $\\langle \\hat{E}^2 \\rangle / 8\\pi$")

if __name__ == "__main__":
    var = sys.argv[1]
    if var == "e2":
        filenames = sys.argv[2:]
        fig, ax = plt.subplots()
        datas = []
        configs = []
        for filename in filenames:
            datas.append(np.loadtxt(filename))
            config_filename = filename[:filename.find("average_E_energy_density")] + "config.pkl"
            try:
                with open(config_filename, 'rb') as file:
                    configs.append(pickle.load(file))
            except:
                print("Failed to open config file!")

        plot_energy(datas, fig, ax, [name[:name.find("average_E_energy_density")] for name in filenames], configs)
        plt.legend()
        plt.show()
    elif var == "et":
        filenames = sys.argv[2:]
        filenames.sort(key = lambda name : int(name[name.find("flds")+5:-4]))
        fig, ax = plt.subplots()
        datas = []
        for filename in filenames:
            datas.append(read_full_box(filename, "ez"))
        plot_in_time(datas, fig, ax)
        plt.show()
    else:
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
