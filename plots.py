import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
import os
import sys

def processipr():
    maxx, maxy, eig = np.loadtxt('paras', dtype=int)
    for i in range(eig):
        iprs = np.zeros((maxx, maxy))
        for x in range( maxx):
            for y in range(maxy):
                cdir = os.getcwd() + '/1tun1cou.{}x.{}y*/ipr'.format(str(x + 1).zfill(2), str(y + 1).zfill(2)) 
                f = glob.glob(cdir)[0]
                iprs[x][y] = np.average(np.loadtxt(f), axis=0)[i]
        np.savetxt('allipr{}'.format(i), iprs)

def allplot(eig):
    maxx, maxy = np.loadtxt('paras', dtype=int)
    iprs = np.loadtxt('allipr{}'.format(eig))
    X, Y = np.meshgrid(list(range(1, maxx + 1)), list(range(1, maxy + 1)))[::-1]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, iprs, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_title('Plotting IPR vs. maximum x and y disorder')
    ax.set_xlabel('Maximum x disorder')
    ax.set_ylabel('Maximum y disorder')
    ax.set_zlabel('IPR')
    
    
    plt.show()

if __name__ == '__main__':
    eig = int(sys.argv[1])
    if os.path.exists('allipr{}'.format(eig)):
        print('plotting {]th eigenvalue')
        allplot(eig)
    else:
        processipr()

