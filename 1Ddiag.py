from gc import collect
import site
import numpy as np
from math import sqrt
from numpy.core.defchararray import multiply
from functools import reduce
import copy
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import multiprocessing as mp
from matplotlib.widgets import Slider
from matplotlib.widgets import TextBox
import os.path
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib as mpl
from scipy.sparse import csr_matrix
#from numpy.linalg import eigh


# The preliminary TN simulation program I have for the small scale 2D system we have. 

# Notations: upper case letter are always the collection, where lower case indicates a member of that collection

# intialize the parameters for the simulation
def initParameters():
    L, num_e, maxcase, lo, hi, mode = np.loadtxt('inp')
    mode = int(mode)
    num_site = 0
    maxlen = 0

    # mode 0 is the universal x, y disorder generation, used in testing a new t, lambda combination
    # mode 1 is select site generation, only on x direction, and the rest of the site have minimal disorder. 
    if mode == 0:
        tun, cou, a, b,  readdisorder, seed, decay, distype = np.loadtxt('para_dis', dtype='str')

    elif mode == 1:
        # the a, b now refers to the lower and upper limit of the site disorder, a sign is assigned randomly.
        tun, cou, a, b,  readdisorder, seed, decay, distype, num_site, maxlen = np.loadtxt('para_dis', dtype='str')

    # mode 2 generates all maxlen cases
    # mode 3 is the interactive mode
    else:
        tun, cou, a, b,  readdisorder, seed, decay, distype, num_site  = np.loadtxt('para_dis', dtype='str')


    t, int_ee, int_ne, z, zeta, ex, selfnuc, hopmode = np.loadtxt('hamiltonian')
    para = {
    'L' : int(L),
    'num_e' : int(num_e),
    't': t,
    'int_ee': int_ee,
    'int_ne': int_ne,
    'z': z,
    'zeta':zeta,
    'ex': ex,
    # if-include-nuc-self-int switch, 1 means include
    'selfnuc': int(selfnuc),
    'tun': int(tun),
    'cou': int(cou),
    'a': float(a),
    'b': float(b),
    'seed': int(seed),
    'readdisorder': int(readdisorder),
    'decay': float(decay),
    'maxcase': int(maxcase),
    'distype': distype,
    'Nth eig': [int(lo), int(hi)],
    # mode controls the generation type. 0 is generation on all sites, 1 is controlled generation on fixed number of sites on singular maxlen, 2 is 1 but on all possible maxlens
    'mode': mode,
    'num_site': int(num_site),
    'maxlen':int(maxlen),
    'hopmode': int(hopmode),
    'sparse':0}
    
    if mode == 1:
        para['batch'] = para['maxcase'] * (para['L'] - para['maxlen'] + 1)

    else:
        para['batch'] = para['maxcase'] * (para['L'] - para['num_site'] + 1)

    print('Simulation parameters: {}'.format(para))
    return para


def generateState(L, num_e):
    if L == num_e:
        return [[1] * num_e]
    if L == 0:
        return [[]]
    if num_e == 0:
        return [[0] * L]
    return [ [0]  + state for state in generateState(L - 1, num_e)] + [ [ 1] + state for state in generateState(L - 1, num_e -1)]

def generateDisorder(para):

    #L, batch, readdisorder, distype, seed , a, b = para['L'],   para['batch'], para['readdisorder'], para['distype'], para['seed'], para['a'], para['b']
    L, maxcase, readdisorder, distype, a, b = para['L'],   para['maxcase'], para['readdisorder'], para['distype'], para['a'], para['b']
    mode = para['mode']
    num_site = para['num_site']
    maxlen = para['maxlen']
    batch = para['batch']

    if readdisorder:
        disx = np.loadtxt('disx')
        disy = np.loadtxt('disy')
        sites = np.loadtxt('sites', dtype=int)

    else:
        rng = np.random.default_rng()
        
        if mode == 0:
            if distype == 'uniform':
                return a * rng.uniform(-1, 1, (batch, L)), b * rng.uniform(-1, 1, (batch, L))

            elif distype =='gaussian':
                return rng.normal(0, a, (batch, L)), rng.normal(0, b, (batch, L))

        else:
            
            sites = []

            if mode == 1:
                # this mode generate target maxlen
                for begin in range(0, L - maxlen + 1):

                    # we first choose a random starting position, after which the disorder will be generated in that region
                    
                    if num_site > 2:

                        for _ in range(maxcase):
                            newstate = np.concatenate( ([begin], sorted(rng.choice(np.arange(begin + 1,  begin + maxlen - 1), num_site - 2, replace=False)), [begin + maxlen -1] ) )
                            sites += [newstate]

                    elif num_site == 2:

                        for _ in range(maxcase):
                            newstate = np.array( [begin, begin + maxlen -1])
                            sites += [newstate]

                    else:

                        for _ in range(maxcase):
                            newstate = np.array ( [begin])
                            sites += [newstate]

                    
            
                print(len(sites))
                #print(sites)

            elif mode == 2:
                # this is the # of cases for each maxlen 
                #maxcase = batch // (L - num_site + 1)
                # case counter for each maxlen case
                casecnt = defaultdict(int)
                caselist = list(range(num_site, L + 1))

                # when there are still maxlen not examined, generate cases of that maxlen
                while caselist:
                    
                    cur_max = caselist[-1]

                    start = rng.choice(L - cur_max + 1, 1)
                    newstate = sorted(rng.choice(np.arange(start, start+cur_max), num_site, replace=False))
                    begin, end = min(newstate), max(newstate)

                    if casecnt[ end - begin + 1] < maxcase:
                        sites += [newstate]
                        casecnt[ end - begin + 1] += 1

                        if casecnt[ end - begin + 1] == maxcase:
                            print('maxlen {} OK'.format(end - begin + 1 ))
                            caselist.remove(end - begin + 1 )

            
            sites = np.array(sites)
            #generate the site positions
            #sites = np.array([ sorted(rng.choice(np.arange(start, start + maxlen), num_site, replace=False)) for _ in range(batch)])

            disx = np.zeros((batch, L))
            disy = np.zeros((batch, L))

            sign = np.array([[ -1 if np.random.random() < 0.5 else 1 for _ in range(num_site)] for _ in range(batch) ])

            newdisx =  rng.uniform(a, b, (batch, num_site))  * sign

            for b in range(batch):
                # make sure that the sites do not cross each other
                site = 0
                while site < num_site - 1:
                    
                    # if nearest neighbor and N - 1 extends past N
                    if sites[b][site] == sites[b][site + 1] - 1 and newdisx [b][ site]  > 1 + newdisx [b][ site + 1] :
                        disx[b][ sites[b][site] ] = 1 + newdisx [b][ site + 1] 
                        disx[b][ sites[b][site + 1] ] = newdisx [b][ site] - 1
                        site += 2

                    else:
                        disx[b][ sites[b][site] ] = newdisx [b][ site]
                        site += 1

                if site < num_site:
                    disx[b][ sites[b][-1] ] = newdisx [b][ -1]


    return disx, disy, sites


def init(para):
    L = para['L']
    num_e = para['num_e']
    states = generateState(L, num_e)
    return states

def initdict(S):
    num_state = len(S)
    L = len(S[0])

    occdict = np.zeros( (num_state, L) )
    balancestate = np.zeros( num_state)
    sdict = {}

    for i, state in enumerate(S):
        sdict[str(state)] = i

        for j, occ in enumerate(state):
            #print(i, j)
            occdict[ i, j ] = occ

        if sum( state[:L//2]) == 3:
            balancestate[i] = 1

    #print(balancestate)
    return sdict, occdict, balancestate

def hamiltonian(s, dis, para):
    L, t, int_ee, int_ne, z, zeta, ex, selfnuc = para['L'],  para['t'], para['int_ee'],para['int_ne'], para['z'], para['zeta'], para['ex'],  para['selfnuc']
    tun, cou, decay = para['tun'], para['cou'], para['decay']
    allnewstates = [[], []]
    allee, allne = 0, 0
    hopmode = para['hopmode']

    def checkHopping(loc):
        # set up the NN matrix
        ts = []
        res = []
        
        #hop right
        if loc < L - 1 and s[loc] != s[loc + 1]:
            snew = copy.copy(s) 
            snew[loc], snew[ loc +1] = snew[ loc + 1], snew[loc]
            res.append(snew)
            if tun:
                dx = - dis[0][loc]  + dis[0][loc + 1] + 1
                dy = - dis[1][loc] + dis[1][loc + 1] 
                dr = sqrt(dx ** 2 + dy ** 2)

                # exponential decay 
                if hopmode == 0:
                    factor = np.exp( - ( dr - 1) /decay )

                else:
                    factor = np.sin( dr )
                #print(factor)
                ts.append( -t * factor)
            else:
                ts.append(- t)

        # sum the hopping terms
        #print(ts)
        return ts, res

    def ee(loc):  
        total_ee = 0
        for site in range(L):
            # no same-site interaction bc of Pauli exclusion
            if site != loc:
                
                # distance between sites
                r = site - loc
                
                # exchange interaction
                if abs(r) == 1:
                    factor = 1 - ex
                
                else:
                    factor = 1

                # Disorder
                if cou:
                    r = sqrt( ( - dis[0][loc] + r + dis[0][site]) ** 2 + ( - dis[1][loc] + dis[1][site] ) ** 2)
                
                # adding contribution to the total contribution
                total_ee +=  int_ee * z * factor / ( abs(r) + zeta ) * s[loc] * s[site]
        return total_ee


    def ne(loc):
        total_ne = 0
        # sum the contribution from all sites
        for site in range(L):

            # distance between sites
            r = site - loc

            # disorder
            if cou:
                r = sqrt( ( - dis[0][loc] + r + dis[0][site]) ** 2 + ( - dis[1][loc] + dis[1][site] ) ** 2)

            total_ne +=  int_ne * z / ( abs(r) + zeta ) * s[site]

        # self nuclear interaction condition
        return total_ne if selfnuc else total_ne - int_ne * z / zeta * s[loc]

    for loc in range(L):

        # the hopping part. Set up the changed basis states
        ts, newstate =  checkHopping(loc)
        for i in range(len(ts)):
            allnewstates[0].append(ts[i])
            allnewstates[1].append(newstate[i])


        # the ee interaction part, the 0.5 is for the double counting of sites. 
        allee += ee(loc) 
        # the ne interaction part
        allne += ne(loc)
        #print(ne(row, col))
        

    #print(allee, allne)

    allnewstates[0].append(allee + allne)

    allnewstates[1].append(s)

    return allnewstates



def setMatrix(S, N, dis, sdict, para):

    sparse = para['sparse'] 

    def checksparce():
        
        cnt = defaultdict(int)
        for row in M:
            cnt[len(np.argwhere(row))] += 1

        print(cnt)

    if sparse:

        row = []
        col = []
        val = []

        for i, state in enumerate(S):
            newstates = hamiltonian(state, dis, para)
            for j, newstate  in enumerate(newstates[1]):
                row += [i]
                col += [sdict[str(newstate)]]
                val += [newstates[0][j]]

        M = csr_matrix((val, (row, col)), shape=(N, N))

    else:
        M = np.zeros((N, N))

        for i, state in enumerate(S):
            newstates = hamiltonian(state, dis, para)
            for j, newstate  in enumerate(newstates[1]):
                M[i][sdict[str(newstate)]] = newstates[0][j]

        checksparce()
    
    return M
    
def solve(M, para):
    ranges = para['Nth eig']
    sparse = para['sparse']
    # for use in sparse solver
    

    #print( check_symmetric(M))
    #print( M )

    start = time.time()

    if sparse:
        k = ranges[-1] - ranges[0] + 1

        print(k)
        w, v = eigs(M, k - 2)

    else:

        print(ranges)
        w, v = eigh(M, subset_by_index=ranges)
    
    print('diag time {}'.format(time.time()- start))

    return w, v
    #return eigh(M)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def plotprob(eigv, para):

    N = len(eigv[0])
    
    for eig in range(N):
        if eig == 1:
            plt.title('Wavefunction Plot')
        plt.subplot(N, 1, eig + 1)
        plt.plot(list(range(len(eigv))), eigv[:,eig] )
    
    plt.show()


def calIPR(eigv):
    return np.sum(eigv**4, axis=0)

def saveresults(allres, allmany_res, site_info, para):

    
    L = para['L']
    lo, hi = para['Nth eig']
    fullrange = hi - lo + 1
    strs = ['energy', 'ipr', 'disx', 'disy']
    num_e = para['num_e']


    allres = np.array(allres)

    disx = allres[ :, : L]
    disy = allres[ :, L : L * 2]
    energy = allres[ :, 2 * L : 2 * L + fullrange]
    ipr = allres[:, 2 * L + fullrange : 2 * L + fullrange * 2]

    data = [energy, ipr, disx, disy]
    
    for i in range(len(strs)):
        np.savetxt(strs[i], data[i], fmt='%.8f')

    if num_e > 1:

        allmany_res = np.array(allmany_res)
        tcd = allmany_res[:, : L * fullrange]
        gpi = allmany_res[ :, L * fullrange : L * fullrange + fullrange]
        balance = allmany_res [ :, L * fullrange + fullrange : L * fullrange + fullrange * 2]

        manydata = [ tcd, gpi, balance]
        tag = ['tcd', 'gpi', 'balance']

        for i in range(len(tag)):
            np.savetxt( tag[i], manydata[i])


    np.savetxt('sites', site_info, fmt='%i')




#begin the many-body related metrics
# GPI




def single_iteration(case, disx, disy, sites, S,  sdict, occdict, balancestate, para):


    def cal_TCD():

        # assume full spectrum
        

        tcd = np.zeros(( k,  L))

        for n in range(k):
            for i in range(L):
                tcd[n, i] = sum(eigv[:, 0] * eigv[:, n] * occdict[:, i])

        return tcd

    def distance(i, j):

        return sqrt((j - i + dis[0][j] - dis[0][i]) ** 2 + ( dis[1][j] - dis[1][i]) ** 2) 

    def ee(i, j):
        
        #ex = 0
        #zeta = 0
        
        if abs(j - i) == 1:
            factor = 1 - ex

        else:
            factor = 1

        return z * int_ee * factor / ( distance(i, j) + zeta)
        

    def cal_GPI():
        gpi = np.zeros(k)

        for n in range(k):
            for i in range(L):
                for j in range(L):
                    
                    
                    gpi[n] += tcd[ n, i] * tcd[n, j] * ee(i, j) 

        #print(gpi)
        return gpi

    def cal_balance():
        return np.array([sum( (balancestate * eigv[:, n]) ** 2) for n in range(k) ])


    
    def allplot():

        fig, ax = plt.subplots(1,4, figsize = (20, 5))
        title = 'Plots for $\lambda$ = {}t, L = {}, N = {}'.format(int_ee, L, num_e)
        internal_plot(fig, ax, tcd, gpi, balance, energy, disx[case], title)

    num_e = para['num_e']
    L = para['L']
    mode = para['mode']

    z, int_ee, zeta, ex = para['z'], para['int_ee'], para['zeta'], para['ex']

    print('\n \n \n case: {}'.format(case))

    dis = [disx[case], disy[case]]
    #print('x Disorder: {}\n y Disorder: {}'.format(dis[0], dis[1]))
    # total number of states
    N = len(S)

    # in full spectrum k == N
    k = para['Nth eig'][-1] - para['Nth eig'][0] + 1

    M = setMatrix(S, N, dis, sdict,  para)

    #print(M)
    energy, eigv = solve(M, para)

    #plotprob(eigv, para)
    #print('Eigenvectors (by column): \n {}'.format(eigv))
    ipr = calIPR(eigv)

    if num_e > 1:
        tcd = cal_TCD()

        gpi = cal_GPI()

        balance = cal_balance()

        
    #print(energy)
    #print(ipr)

    if mode < 3 :
        #allplot()

        res = np.concatenate( (dis[0], dis[1], energy, ipr))
        site_res = sites[case]

        if num_e > 1:
            many_res = np.concatenate( (tcd.reshape( k * L), gpi, balance))
            print(len(many_res))

        else:
            many_res = []

        #print(len(res))
        return [res, many_res, site_res]
        #collect_result(energy, ipr)

        #print('Energy is {}'.format(energy))
        #print('Inverse participation ratio: {}'.format(ipr))

    else:
        return tcd, gpi, balance, energy


def internal_plot(fig, ax, tcd, gpi, balance, energy, dis, title):
    
    L = len(tcd[0])
    N = len(balance)

    for i in range(4):
        ax[i].clear()

    gpi = np.abs(gpi[1:])
    gs = energy[0]
    plasmon = energy[1] - gs

    energy = energy[1:]

    idx = np.argsort( gpi)[::-1][:6]


    ax[0].plot( np.arange(1, L + 1), tcd[1, :])
    ax[0].plot( np.arange(0.5, L + 1.5), [tcd[1, 0] / 2] + [ (tcd[1, j] + tcd[1, j - 1])/2 for j in range(1, L)] + [tcd[1, -1]/2])
    ax[1].scatter ( [ ( ene - gs) / (plasmon) for ene in np.delete(energy, idx)], np.delete(gpi, idx))
    ax[1].scatter ( [ ( ene - gs) / (plasmon) for ene in energy[idx]], gpi[idx], c = 'red')
    ax[1].set_xlim(0, 20)
    ax[2].scatter( np.arange(N), balance, s = 3)
    ax[3].scatter( np.arange(1, L + 1) + dis, np.ones(L))
    ax[3].set_ylim(0, 2)
    ax[3].set_xlim(0, L + 1)
    ax[3].set_axis_off()

    ax[0].set_title('TCD vs site')
    ax[1].set_title('GPI vs excitation energy/ plasmon energy')
    ax[2].set_title('Balance vs eigenstate')
    ax[3].set_title('Visualization of disorder')

    ax[0].set_xlabel('site number')
    ax[1].set_xlabel('Excitation energy/ plasmon energy')
    ax[2].set_xlabel('eigenstate number')

    fig.suptitle(title)
    #plt.show()

def interactive(S, sdict, occdict, balancestate, para):


    case = 0
    L = para['L']
    int_ee = para['int_ee']
    N = len(S)
    slide = 0
    mode = para['mode']

    total = 4
    fig, ax = plt.subplots(1,total, figsize = (total * 5, 5))

    if mode == 3:

        if slide == 1:
            loc = Slider(plt.axes([0.25, 0.05, 0.5, 0.02]), 'location', 0, L, valstep=1)
            dis = Slider(plt.axes([0.25, 0.0, 0.5, 0.02]), 'disorder', -1, 1 , valstep=0.01)

        else:
            loc = TextBox(plt.axes([0.25, 0.05, 0.5, 0.02]), 'location')
            dis = TextBox(plt.axes([0.25, 0.0, 0.5, 0.02]), 'disorder')

        
        disy = np.zeros((1, L))

        def update(val):

            disx = np.zeros((1, L))

            if slide == 1:
                site = int(loc.val)
                mag = dis.val

            else:
                site = int(loc.text)
                mag = float(dis.text)

            disx[0][site] = mag

            tcd, gpi, balance, energy = single_iteration(case, disx, disy, site, S,  sdict, occdict, balancestate, para)
            
            title = 'Plots for $\lambda$ = {}t, L = {}, N = {}'.format(int_ee, L, num_e)
            internal_plot(fig, ax, tcd, gpi, balance, energy, disx[0], title)



        if slide == 1:
            loc.on_changed(update)
            dis.on_changed(update)

        else:
            loc.on_submit(update)
            dis.on_submit(update)

        plt.show()

    else:
        if os.path.exists('tcd') and os.path.exists('disx') and os.path.exists('sites'):
            
            def ee(i, j):
                
                ex, z, zeta = para['ex'], para['z'], zeta['zeta']
                #ex = 0
                #zeta = 0
                
                if j - i == 1:
                    factor = 1 - ex

                else:
                    factor = 1

                return z * int_ee * factor / ( abs(j - i) + zeta)

            sites = np.loadtxt('sites', dtype=int)
            disx = np.loadtxt('disx')

            
            #print(idx)
            

            discard = []
            for i, dis in enumerate(disx):

                if np.count_nonzero(dis) == 2:
                    one, two = np.nonzero(dis)[0]
                    if one == two - 1 and dis[one] >= dis[two] + 1:
                        discard += [i]

            print(len(discard))
            sites = np.delete(sites, discard, axis=0)

            idx = [ comb[0] for comb in sorted( [ (i, val) for i, val in enumerate(sites)], key = lambda x: [ x[1][0], disx[ x[0] ][ x[1][0] ], x[1][1],  disx[ x[0] ][ x[1][1]]   ])]

            sites = np.delete(sites, discard, axis =0 )[idx]
            disx = np.delete(disx, discard, axis=0)[idx]

            
            #idx = np.array([y[1] for y in sorted( [(val, i) for i, val in enumerate(disx)], key=lambda x: [x[0][i] for i in range(L)])])

                
            tcd = np.delete(np.loadtxt('tcd'), discard, axis= 0)[idx]

            if not os.path.exists('gpi'):
                cases = len(tcd)
                tcd = tcd.reshape((cases, 924, 12))
                gpi = np.zeros((cases, 924))
                for case in range(cases):
                    for n in range(924):
                        for i in range(L):
                            for j in range(L):
                                gpi[case] += tcd[case, n, i] * tcd[case, n, j] *  ee(i, j) 

                np.savetxt('gpi', gpi)

            gpi = np.delete(np.loadtxt('gpi'), discard, axis= 0)[idx]
            energy = np.delete(np.loadtxt('energy'), discard, axis = 0)[idx]

            balance = np.delete(np.loadtxt('balance'), discard, axis =0)[idx]
            case = len(tcd)

            tcd = tcd.reshape((case, N, L))
            

            for row in range(len(tcd)):
                for n in range(N):
                    
                    '''
                    idx = np.argmax( np.abs(tcd[row, n, :]))
                    if abs(tcd[row - 1, n, idx] - tcd[row, n, idx]) > np.abs(tcd[row, n, idx]):

                        tcd[row, n, :] = -tcd[row, n, :]
                    ''' 
                    if tcd[row, n, 0] > 0:
                        tcd[row, n, :] = - tcd[row, n, :]

            

            mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\Ray\\Downloads\\ffmpeg-master-latest-win64-gpl\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe'

            lo = np.amin( tcd[:, 1, :])
            hi = np.amax( tcd[:, 1, :])

            def animate(i):
                
                curtcd = tcd[i, :, :]
                curgpi = gpi[i, :]
                curbalance = balance[i, :]
                curenergy = energy[i, :]

                dis = disx[i]

                #pos1, pos2 = ref[ i // (19 * 19)]

                #val1 = dis[pos1]
                #val2 = dis[pos2]
                
                print('frame {}'.format(i))
                title = '2 site disorder scan, site 1 at position {} with disorder {}, site 2 at position {} with disorder {}'.format(sites[i][0] + 1, disx[i][sites[i][0] ], sites[i][1] + 1, disx[i][ sites[i][1]])
                internal_plot(fig, ax, curtcd, curgpi, curbalance, curenergy, dis, title, lo, hi)

            cnt = 0
            ref = {}

            for i in range(L):
                for j in range(i + 1, L):
                    ref[ cnt ] = (i, j)
                    cnt += 1


            anim = FuncAnimation(fig, animate, frames= case, interval=200,  repeat=False)

            #plt.show()
            writervideo = animation.FFMpegWriter(fps=5)
            anim.save( 'timelapse.mp4', writer=writervideo)

        else:

            print('Data missing')



if __name__ == '__main__':

    para = initParameters()
    num_e = para['num_e']
    readdisorder = para['readdisorder']
    mode = para['mode']
    # Start the iterative Monte Carlo updates
    energy = []

    # Here we try using a randomly generated set of occupation configuration
    S = init(para)

    # generate dictionary for book keeping
    sdict, occdict, balancestate = initdict(S)

    #print(occdict.shape)

    if mode < 3:
        disx, disy, sites = generateDisorder(para) 
        
        #cases = 1
        cases = len(disx)

        allres = []
        allmany_res = []
        site_info = []

        start_time = time.time()
        pool = mp.Pool(mp.cpu_count())

        def collect_result( result):

            result, many_result, site_res = result

            if num_e > 1:
                allmany_res.append(many_result)

            allres.append(result)
            site_info.append(site_res)


        for case in range(cases):
            
            x = pool.apply_async(single_iteration, args=(case, disx, disy, sites, S, sdict, occdict, balancestate, para), callback=collect_result)
        
        x.get()
        pool.close()
        pool.join()
            
        # the order is disx, disy, energy, ipr
        saveresults(allres, allmany_res, site_info, para)
        print('finish time: {}'.format(time.time() - start_time))
            #calEnergy(S, A, para)
            #S = initSpin(rdim, cdim)
            #print(hamiltonian(S, rdim, cdim, t, int_ee, int_ne, Z, zeta, ex))

    # interactive mode
    else:
        interactive(S, sdict, occdict, balancestate, para)
        
