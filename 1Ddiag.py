import numpy as np
from math import sqrt
from numpy.core.defchararray import multiply
from functools import reduce
import copy
from scipy import linalg
from scipy.linalg import eigh
import matplotlib.pyplot as plt


# The preliminary TN simulation program I have for the small scale 2D system we have. 

# Notations: upper case letter are always the collection, where lower case indicates a member of that collection

# intialize the parameters for the simulation
def initParameters():
    L, N, batch, lo, hi = np.loadtxt('inp')
    tun, cou, a, b, readdisorder, seed, decay = np.loadtxt('para_dis')
    t, int_ee, int_ne, z, zeta, ex, selfnuc = np.loadtxt('hamiltonian')
    para = {
    'L' : int(L),
    'N' : int(N),
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
    'a': a,
    'b': b,
    'readdisorder': int(readdisorder),
    'seed': int(seed),
    'decay': decay,
    'batch': int(batch),
    'Nth eig': [int(lo), int(hi)]}
    print('Simulation parameters: {}'.format(para))
    return para


def generateState(L, N):
    if L == N:
        return [[1] * N]
    if L == 0:
        return [[]]
    if N == 0:
        return [[0] * L]
    return [ [0]  + state for state in generateState(L - 1, N)] + [ [ 1] + state for state in generateState(L - 1, N -1)]

def generateDisorder(para):
    L, a, b, batch, readdisorder = para['L'],  para['a'], para['b'], para['batch'], para['readdisorder']
    seed = para['seed']

    #print(seed)
    if readdisorder:
        disx, disy = np.loadtxt('val_dis')
        return disx, disy
    else:
        rng = np.random.default_rng(seed=seed)

        return a * rng.uniform(-1, 1, (batch, L)), b * rng.uniform(-1, 1, (batch, L))

def init(para):
    L = para['L']
    N = para['N']
    states = generateState(L, N)
    return states

def initdict(S):
    sdict = {}
    for i, state in enumerate(S):
        sdict[str(state)] = i
    return sdict 

def hamiltonian(s, dis, para):
    L, t, int_ee, int_ne, z, zeta, ex, selfnuc = para['L'],  para['t'], para['int_ee'],para['int_ne'], para['z'], para['zeta'], para['ex'],  para['selfnuc']
    tun, cou, decay = para['tun'], para['cou'], para['decay']
    allnewstates = [[], []]
    allee, allne = 0, 0

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
                factor = np.exp( - ( dr - 1) /decay )
                #print(factor)
                ts.append( -t * factor)
            else:
                ts.append(- t)

        # sum the hopping terms
        #print(ts)
        return ts, res

    def ee(loc):  
        res = 0
        for site in range(L):
            # no self-interaction
            if site != loc:

                r = site - loc
                factor = [ 1 - ex if np.rint(abs(r)) == 1 else 1][0]

                if cou:
                    r = sqrt( ( - dis[0][loc] + r + dis[0][site]) ** 2 + ( - dis[1][loc] + dis[1][site] ) ** 2)
                # check exchange condition
                
                
                res +=  int_ee * z * factor / ( abs(r) + zeta ) * s[loc] * s[site]
        return res


    def ne(loc):
        res = 0
        # sum the contribution from all sites
        for site in range(L):
            r = site - loc

            if cou:
                r = sqrt( ( - dis[0][loc] + r + dis[0][site]) ** 2 + ( - dis[1][loc] + dis[1][site] ) ** 2)

            res +=  int_ne * z / ( abs(r) + zeta ) * s[site]
        return res if selfnuc else res - int_ne * z / zeta * s[loc]

    for loc in range(L):

        # the hopping part
        ts, newstate =  checkHopping(loc)
        for i in range(len(ts)):
            allnewstates[0].append(ts[i])
            allnewstates[1].append(newstate[i])


        # the ee interaction part, the 0.5 is for the double counting of sites. 
        allee += ee(loc) * 0.5
        # the ne interaction part

        #print(ne(row, col))
        allne += ne(loc)

    #print(allee, allne)

    allnewstates[0].append(allee + allne)

    allnewstates[1].append(s)

    return allnewstates



def setMatrix(S, N, dis, sdict, para):
    M = np.zeros((N, N))
    for i, state in enumerate(S):
        newstates = hamiltonian(state, dis, para)
        for j, newstate  in enumerate(newstates[1]):
            M[i][sdict[str(newstate)]] = newstates[0][j]
    return M
    
def solve(M, para):
    ranges = para['Nth eig']
    return eigh(M, subset_by_index=ranges)

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

def saveresults(energy, ipr, disx, disy):
    strs = ['energy', 'ipr', 'disx', 'disy']
    data = [energy, ipr, disx, disy]
    
    for i in range(len(strs)):
        np.savetxt(strs[i], data[i], fmt='%.8f')

if __name__ == '__main__':
    para = initParameters()
    batch = para['batch']
    readdisorder = para['readdisorder']
    # Start the iterative Monte Carlo updates
    energy = []

    # Here we try using a randomly generated set of occupation configuration
    S = init(para)

    # generate dictionary for book keeping
    sdict = initdict(S)
    disx, disy = generateDisorder(para) 

    if readdisorder :
        batch = 1
        disx, disy = [disx], [disy]

    energies = []
    iprs = []

    for case in range(batch):
        # generate disorder
        
        print('\n \n \n case: {}'.format(case))

        dis = [disx[case], disy[case]]
        print('x Disorder: {}\n y Disorder: {}'.format(dis[0], dis[1]))
        # total number of states
        N = len(S)

        M = setMatrix(S, N, dis, sdict, para)

        #print(M)
        energy, eigv = solve(M, para)

        #plotprob(eigv, para)
        #print('Eigenvectors (by column): \n {}'.format(eigv))
        ipr = calIPR(eigv)

        energies.append(energy)
        iprs.append(ipr)
        print('Energy is {}'.format(energy))
        print('Inverse participation ratio: {}'.format(ipr))

    saveresults(energies, iprs, disx, disy)
        #calEnergy(S, A, para)
        #S = initSpin(rdim, cdim)
        #print(hamiltonian(S, rdim, cdim, t, int_ee, int_ne, Z, zeta, ex))
