import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
import os
import sys


def loadpara():
    maxx, maxy, numeig , eig = np.loadtxt('paras', dtype=int)
    para = {
        'maxx': maxx,
        'maxy': maxy,
        'numeig': numeig,
        'whichEig': eig,
    }
    return para

def processipr(para):
    maxx, maxy, numeig = para['maxx'], para['maxy'], para['numeig']

    for i in range(numeig):
        iprs = np.zeros((maxx, maxy))
        for x in range( maxx):
            for y in range(maxy):
                cdir = os.getcwd() + '/1tun1cou.{}x.{}y*/ipr'.format(str(x + 1).zfill(2), str(y + 1).zfill(2)) 
                f = glob.glob(cdir)[0]
                iprs[x][y] = np.average(np.loadtxt(f), axis=0)[i]
        np.savetxt('allipr{}'.format(i), iprs)

def allplot(para):
    maxx, maxy, eig = para['maxx'], para['maxy'], para['whichEig']

    iprs = np.loadtxt('allipr{}'.format(eig))
    X, Y = np.meshgrid(list(range(1, maxx + 1)), list(range(1, maxy + 1)))[::-1]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, iprs, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_title('Plotting IPR vs. maximum x and y disorder, eigenstate {}'.format(eig))
    ax.set_xlabel('Maximum x disorder')
    ax.set_ylabel('Maximum y disorder')
    ax.set_zlabel('IPR')

    
    plt.show()

if __name__ == '__main__':
    para = loadpara()

    if os.path.exists('allipr{}'.format(para['whichEig'])):
        print('plotting {]th eigenstate'.format(para['whichEig']))
        allplot(para)
    else:
        print('processing data')
        processipr(para)

