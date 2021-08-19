import numpy as np
from math import sqrt
from numpy.core.defchararray import multiply
from functools import reduce
import copy
from scipy import linalg
from scipy.linalg import eigh


# The preliminary TN simulation program I have for the small scale 2D system we have. 

# Notations: upper case letter are always the collection, where lower case indicates a member of that collection

# intialize the parameters for the simulation
def initParameters():
    L, N, tun, cou, a, b = np.loadtxt('inp')
    para = {
    'L' : int(L),
    'N' : int(N),
    't': 1.0,
    'int_ee': 1,
    'int_ne': -1,
    'z': 1,
    'zeta':0.5,
    'ex': 0.2,
    'tun': int(tun),
    'cou': int(cou),
    'a': a,
    'b': b,
    # if-include-nuc-self-int switch, 1 means include
    'batch': 200,
    'selfnuc':0}
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
    L, N, a, b = para['L'], para['N'], para['a'], para['b']

    return a * np.random.uniform(-1, 1, L) + b * np.random.uniform(-1, 1, L)

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
    tun, cou, a, b = para['tun'], para['cou'], para['a'], para['b']
    allnewstates = [[], []]
    allee, allne = 0, 0

    def checkHopping(loc):
        # set up the NN matrix
        ts = []
        res = []
        
        #hop right
        if not loc == L - 1:
            snew = copy.copy(s) 
            snew[loc], snew[ loc +1] = snew[ loc + 1], snew[loc]
            res.append(snew)
            if tun:
                factor = np.exp( dis[loc] + dis[loc + 1] )
                ts.append( -t * factor)
            else:
                ts.append(- t)

        # sum the hopping terms
        #print(ts)
        return ts, res

    def ee(loc):  
        res = 0
        for site in range(L):
            r = abs(site - loc)
            if cou:
                r += sum(dis[min(loc, site): max(loc, site) + 1])
            # check exchange condition
            factor = [ 1 - ex if np.rint(r) == 1 else 1][0]
            # remove self-interaction
            if site != loc :
                res +=  int_ee * z * factor / ( r + zeta ) * s[loc] * s[site]
        return res


    def ne(loc):
        res = 0
        # sum the contribution from all sites
        for site in range(L):
            r = abs(site - loc)

            if cou:
                r += sum(dis[min(loc, site): max(loc, site) + 1])

            res +=  int_ne * z / ( r + zeta ) * s[site]
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

# B is the altered state
def innerProduct(A, B):
    return sum([B[0][i] if np.array_equal(A, s) else 0 for i, s in enumerate(B[1]) ])


def setMatrix(S, N, dis, sdict, para):
    M = np.zeros((N, N))
    for i, state in enumerate(S):
        newstates = hamiltonian(state, dis, para)
        for j, newstate  in enumerate(newstates[1]):
            M[i][sdict[str(newstate)]] = newstates[0][j]
    return M
def solve(M):
    return eigh(M, subset_by_index=[0, 5])

def calIPR(eigv):
    return np.sum(eigv**4, axis=0)

if __name__ == '__main__':
    para = initParameters()
    batch = para['batch']
    # Start the iterative Monte Carlo updates
    energy = []

    # Here we try using a randomly generated set of occupation configuration
    S = init(para)
    # generate dictionary for book keeping
    sdict = initdict(S)

    for step in range(batch):
        # generate disorder
        dis = generateDisorder(para)
        print('Disorder: {}'.format(dis))
        # total number of states
        N = len(S)

        M = setMatrix(S, N, dis, sdict, para)
        #print(M)
        energy, eigv = solve(M)

        ipr = calIPR(eigv)
        print('energy is {}'.format(energy))
        print('inverse participation ratio: {}'.format(ipr))
        #calEnergy(S, A, para)
        #S = initSpin(rdim, cdim)
        #print(hamiltonian(S, rdim, cdim, t, int_ee, int_ne, Z, zeta, ex))
