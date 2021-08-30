import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
import os


def allplot():
    maxx, maxy = np.loadtxt('paras', dtype=int)
    iprs = []
    for x in range(1, maxx):
        for y in range(1, maxy):
            cdir = os.getcwd() + '/1tun1cou{}x{}y*/inp'.format(str(x*0.01)[1:], str(y*0.01)[1:]) 
            f = glob.glob(cdir)
            iprs.append(np.average(np.loadtxt(f), axis=0)[0])
    X, Y = np.meshgrid(list(range(1, maxx)), list(range(1, maxy)))[::-1]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, iprs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    

if __name__ == '__main__':
    allplot()

