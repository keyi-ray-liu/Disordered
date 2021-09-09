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
    rdim, cdim, N, batch, lo, hi = np.loadtxt('2Dinp')
    tun, cou, a, b,  readdisorder, seed, decay, distype = np.loadtxt('para_dis', dtype='str')
    t, int_ee, int_ne, z, zeta, ex, selfnuc = np.loadtxt('hamiltonian')
    para = {
    'rdim': int(rdim),
    'cdim': int(cdim),
    'L': int(rdim) * int(cdim),
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
    'a': float(a),
    'b': float(b),
    'seed': int(seed),
    'readdisorder': int(readdisorder),
    'decay': float(decay),
    'batch': int(batch),
    'distype': distype,
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
    rdim, cdim, batch, readdisorder, distype, seed , a, b = para['rdim'], para['cdim'], para['batch'], para['readdisorder'], para['distype'], para['seed'], para['a'], para['b']

    if readdisorder:
        disx, disy = np.loadtxt('val_dis')
        return disx, disy

    else:
        rng = np.random.default_rng(seed=seed)
        orix = np.array( [[list(range(cdim))] * rdim ] * batch )
        oriy = np.array( [[[i] * cdim for i in range(rdim)]] * batch )

        if distype == 'uniform':

            return a * rng.uniform(-1, 1, (batch, rdim, cdim)) + orix , b * rng.uniform(-1, 1, (batch, rdim, cdim)) + oriy

        elif distype =='gaussian':

            return rng.normal(0, a, (batch, rdim, cdim)) + orix , rng.normal(0, b, (batch, rdim, cdim)) + oriy



def init(para):
    L = para['L']
    N = para['N']
    rdim = para['rdim']
    cdim = para['cdim']
    states = generateState(L, N)
    return [np.array(s).reshape((rdim, cdim)) for s in states]

def initdict(S):
    sdict = {}
    for i, state in enumerate(S):
        sdict[str(state)] = i
    return sdict 

def hamiltonian(s, dis, para):
    rdim, cdim , t, int_ee, int_ne, z, zeta, ex, selfnuc = para['rdim'], para['cdim'], para['t'], para['int_ee'],para['int_ne'], para['z'], para['zeta'], para['ex'],  para['selfnuc']
    tun, cou, decay = para['tun'], para['cou'], para['decay']
    allnewstates = [[], []]
    allee, allne = 0, 0

    def checkHopping(row, col):
        # set up the NN matrix
        ts = []
        res = []
        # hop down
        if row < rdim - 1 and s[row][col] != s[row + 1][col]:
            snew = copy.copy(s) 
            snew[row][col], snew[ row + 1][col] = snew[row + 1 ][col], snew[row][col]
            res.append(snew)

            factor = 1
            if tun:
                dx = - dis[0][row][col]  + dis[0][row + 1][col] 
                dy = - dis[1][row][col] + dis[1][row + 1][col] 
                dr = sqrt(dx ** 2 + dy ** 2)
                factor = np.exp( - ( dr - 1) /decay )

            if (list(s[row][col + 1: ]) + list(s[row + 1][ : col] ) ).count(1) % 2:
                ts.append(t * factor)
            else:
                ts.append(-t * factor)

        #hop right
        if col < cdim -1 and s[row][col] != s[row][col + 1]:
            snew = copy.copy(s) 
            snew[row][col], snew[ row ][col +1] = snew[row  ][col + 1], snew[row][col]
            res.append(snew)
            if tun:
                dx = - dis[0][row][col]  + dis[0][row][col + 1] 
                dy = - dis[1][row][col] + dis[1][row][col + 1]
                dr = sqrt(dx ** 2 + dy ** 2)
                factor = np.exp( - ( dr - 1) /decay )
                #print(factor)
                ts.append( -t * factor)
            else:
                ts.append(- t)

        # sum the hopping terms
        #print(ts)
        return ts, res

    def ee(row, col):  
        res = 0
        for srow in range(rdim):
            for scol in range(cdim):
            # no self-interaction
                if srow != row or scol != col:

                    x = srow - row
                    y = scol - col

                    factor = [ 1 - ex if np.rint(sqrt(x**2 + y**2)) == 1 else 1][0]

                    if cou:
                        r = sqrt( ( - dis[0][row][col] + dis[0][srow][scol]) ** 2 + ( - dis[1][row][col] +  dis[1][srow][scol] ) ** 2)
                    # check exchange condition
                
                
                    res +=  int_ee * z * factor / ( abs(r) + zeta ) * s[row][col] * s[srow][scol]
        return res


    def ne(row, col):
        res = 0
        # sum the contribution from all sites
        for srow in range(rdim):
            for scol in range(cdim):


                if cou:
                    r = sqrt( ( - dis[0][row][col] + dis[0][srow][scol]) ** 2 + ( - dis[1][row][col] +  dis[1][srow][scol] ) ** 2)

                res +=  int_ne * z / ( abs(r) + zeta ) * s[srow][scol]

        return res if selfnuc else res - int_ne * z / zeta * s[row][col]

    for row in range(rdim):
        for col in range(cdim):

            # the hopping part
            ts, newstate =  checkHopping(row, col)
            for i in range(len(ts)):
                allnewstates[0].append(ts[i])
                allnewstates[1].append(newstate[i])


            # the ee interaction part, the 0.5 is for the double counting of sites. 
            allee += ee(row, col) * 0.5
            # the ne interaction part

            #print(ne(rdim, cdim))
            allne += ne(row, col)

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

def plotprob(eigv):

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
    disx = [s.flatten() for s in disx]
    disy = [s.flatten() for s in disy]
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
