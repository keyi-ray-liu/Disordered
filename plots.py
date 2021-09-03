import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
import os
import sys
from matplotlib.widgets import TextBox
from math import ceil


def loadpara():
    maxx, maxy, L, numeig , eig, step, distype = np.loadtxt('paras', dtype=str)
    para = {
        'L': int(L),
        'maxx': int(maxx),
        'maxy': int(maxy),
        'numeig': int(numeig),
        'whichEig': int(eig),
        'step':float(step),
        'distype': distype
    }
    return para

def processipr(para):
    maxx, maxy, numeig = para['maxx'], para['maxy'], para['numeig']
    step = para['step']
    factor = ceil(np.log10(1/step))

    for i in range(numeig):
        iprs = np.zeros((maxx, maxy))
        for x in range( maxx):
            ix = int ( (x + 1) * step * 10 ** factor )
            for y in range(maxy):
                iy = int ( (y + 1) * step * 10 ** factor )
                cdir = os.getcwd() + '/1tun1cou.{}x.{}y*/ipr'.format(str(ix).zfill(factor), str(iy).zfill(factor)) 
                f = glob.glob(cdir)[0]
                iprs[x][y] = np.average(np.loadtxt(f), axis=0)[i]
        np.savetxt('allipr{}'.format(i), iprs)

def iprplot(para):
    maxx, maxy, eig = para['maxx'], para['maxy'], para['whichEig']
    step = para['step']
    distype = para['distype']
    
    X, Y = np.meshgrid(list(range(1, maxx + 1)), list(range(1, maxy + 1)))[::-1]

    axes = plt.axes([0.5, 0.3, 0.3, 0.1])
    txteig = TextBox(axes, 'Enter eigenstate')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    def update(val):
        ax.clear()
        eig = int(txteig.text)
        iprs = np.loadtxt('allipr{}'.format(eig))
        
        
        ax.plot_surface(X, Y, iprs, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        ax.set_title('Plotting IPR vs. maximum x and y disorder, eigenstate {}, step {}, {}'.format(eig, step, distype ))
        ax.set_xlabel('Maximum x disorder')
        ax.set_ylabel('Maximum y disorder')
        ax.set_zlabel('IPR')

        fig.canvas.draw_idle()

    txteig.on_submit(update)
    
    plt.show()

def plotdisorder(para):
    maxx, maxy, L = para['maxx'], para['maxy'], para['L']
    step = para['step']
    distype = para['distype']

    factor = ceil(np.log10(1/step))
    
    axes = [plt.axes([0.5, 0.2 ,0.3, 0.09]), plt.axes([0.5, 0.3 ,0.3, 0.09])]

    txtx = TextBox(axes[0], 'Max x disorder step (from {} to {}), with step length {}:'.format(1, maxx, step), initial=1)
    txty = TextBox(axes[1], 'Max y disorder step (from {} to {}), with step length {}:'.format(1, maxy, step), initial=1)

    fig, ax = plt.subplots()

    def submit(val):
        ax.clear()
        x = int ( int(txtx.text) * step * 10 ** factor) 
        y = int ( int(txty.text) * step * 10 ** factor)

        xdir = os.getcwd() + '/1tun1cou.{}x.{}y*/disx'.format(str(x).zfill(factor), str(y).zfill(factor)) 
        ydir = os.getcwd() + '/1tun1cou.{}x.{}y*/disy'.format(str(x).zfill(factor), str(y).zfill(factor)) 
        xf = glob.glob(xdir)[0]
        yf = glob.glob(ydir)[0]

        disx = np.loadtxt(xf)
        disy = np.loadtxt(yf)

        num = len(disx)
        #define color steps
        cstep = 1/num

        for i in range(num):
            # varying colors for each case
            color = (cstep * i, cstep * i, cstep * i)
            eachdisx = disx[i] + list(range(L))
            eachdisy = disy[i]
        
            ax.scatter(eachdisx, eachdisy, s=0.2, color=color)

        ax.set_xlim( 0 - maxx * step , L - 1 + maxy * step)
        ax.set_ylim(-maxy * step, maxy * step)
        ax.set_title('Visualization of Disorder, {}, max a: {}, max b: {}'.format(distype, int(txtx.text) * step, int(txty.text) * step))

        fig.canvas.draw_idle()
        
    txtx.on_submit(submit)
    txty.on_submit(submit)

    plt.show()

if __name__ == '__main__':
    para = loadpara()

    if os.path.exists('allipr{}'.format(para['whichEig'])):
        print('plotting {}th eigenstate'.format(para['whichEig']))
        plotdisorder(para)
        iprplot(para)
        
    else:
        print('processing data')
        processipr(para)

    
