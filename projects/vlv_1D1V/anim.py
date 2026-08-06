import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation
from runko.mpiio_reader import read_field_snapshot
import pickle
import runko


# read simulation output file and reshape to python format
def read_full_box(path, var_name):
    return read_field_snapshot(path)[var_name]


if __name__ == "__main__":
    filenames = sys.argv[1:] 
    filenames.sort(key = lambda name : int(name[name.find("flds")+5:-4]))

    fig, ax = plt.subplots()
    config_filename = filenames[0][:filenames[0].find("flds")] + "config.pkl"
    config = None
    try:
        with open(config_filename, 'rb') as file:
            config = pickle.load(file)
    except:
        print("Failed to open config file!")

    deltaX = 1 / config.skin_depth


    datas = []
    for filename in filenames:
        datas.append(read_full_box(filename, "ez")[:,1,1])

    x_data = np.array(range(len(datas[0]))) * deltaX
    im = ax.plot(x_data, datas[0])[0]
    ax.set_xlabel("Paikka ($d_s$)")
    ax.set_ylabel("Sähkökentän energia $\\langle \\hat{E}^2 \\rangle / 8\\pi$")

    mins = [min(datas[0])]
    maxs = [max(datas[0])]

    def update(frame):
        mins.append(min(datas[frame]) * 1.5)
        maxs.append(max(datas[frame]) * 1.5)
        if len(mins) > 10:
                    mins.pop(0)
                    maxs.pop(0)
        im.set_ydata(datas[frame])
        ax.set_ylim(sum(mins)/len(mins), sum(maxs)/len(maxs))

    anim = animation.FuncAnimation(fig, update, len(datas))

    plt.show()