import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
import os


def allplot():
    maxx, maxy = np.loadtxt('paras', dtype=int)
    iprs = np.zeros((maxx, maxy))
    for x in range( maxx):
        for y in range(maxy):
            cdir = os.getcwd() + '/1tun1cou.{}x.{}y*/ipr'.format(str(x + 1).zfill(2), str(y + 1).zfill(2)) 
            f = glob.glob(cdir)[0]
            iprs[x][y] = np.average(np.loadtxt(f), axis=0)[0]
    X, Y = np.meshgrid(list(range(1, maxx + 1)), list(range(1, maxy + 1)))[::-1]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, iprs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    

if __name__ == '__main__':
    allplot()

