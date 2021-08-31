import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
import os
import sys
from matplotlib.widgets import TextBox


def loadpara():
    maxx, maxy, L, numeig , eig = np.loadtxt('paras', dtype=int)
    para = {
        'L': L,
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

def plotdisorder(para):
    maxx, maxy, L = para['maxx'], para['maxy'], para['L']
    axes = [plt.axes([0.25, 0.2 ,0.3, 0.09]), plt.axes([0.25, 0.3 ,0.3, 0.09])]

    txtx = TextBox(axes[0], 'Max x disorder (from 0.01 to {})'.format(0.01*maxx), initial=1)
    txty = TextBox(axes[1], 'Max y disorder (from 0.01 to {})'.format(0.01*maxy), initial=1)

    def submit(val):
        x = int ( float(txtx.text) * 100) 
        y = int ( float( txty.text) * 100)

        xdir = os.getcwd() + '/1tun1cou.{}x.{}y*/disx'.format(str(x).zfill(2), str(y).zfill(2)) 
        ydir = os.getcwd() + '/1tun1cou.{}x.{}y*/disy'.format(str(x).zfill(2), str(y).zfill(2)) 
        xf = glob.glob(xdir)[0]
        yf = glob.glob(ydir)[0]

        disx = np.loadtxt(xf)
        disy = np.loadtxt(yf)

        disx = [dis + list(range(L)) for dis in disx].flatten()
        disy = disy.flatten()

        plt.scatter(x, y)
        
    txtx.on_changed(submit)
    txty.on_changed(submit)

    plt.show()

if __name__ == '__main__':
    para = loadpara()

    plotdisorder(para)

    if os.path.exists('allipr{}'.format(para['whichEig'])):
        print('plotting {}th eigenstate'.format(para['whichEig']))
        allplot(para)
    else:
        print('processing data')
        processipr(para)

