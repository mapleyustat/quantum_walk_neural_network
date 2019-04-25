
###########################################################################################################################
# Title: Learning a Path on a Graph by Quantum Walks (Quantum Walk Neural Network: QWNN)
# Date: 04/25/2019 - Present
# Author: Minwoo Bae (minwoo.bae@uconn.edu)
# Institute: The Department of Computer Science and Engineering, UCONN
###########################################################################################################################
import numpy as np
#kron: Kronecker product (kron): matrix tensor matrix
from numpy import sqrt, dot, outer, reshape, kron, append, insert, sum, matmul, add
from numpy import transpose as T
from numpy import tensordot as tensor
from numpy.linalg import inv
from numpy.linalg import norm
from numpy import array as vec
from numpy import eye as id
import heatmaps as ht
import matplotlib
import matplotlib.pyplot as plt

# Hadamard operator:
Hm = (1/sqrt(2))*vec([[1, 1],[1, -1]])

# Coin operator (Hm tensor Hm):
C = kron(Hm,Hm)

# COIN space:
Cs = vec([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

# Position space:
Ps = vec([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])

deg =  len(Cs)
N = len(Ps)

def unitary_operator(time):

    temp = []
    S_temp = []
    t = time

    for n in range(N):
        for i in range(deg):
            n_up = (n+(i+1))%N
            temp = kron(outer(Cs[i], Cs[i]),outer(Ps[n_up], Ps[n]))
            S_temp.append(temp)

    S = S_temp[0]
    for j in range(1,len(S_temp)):
        S = add(S, S_temp[j])

    I = id(N, dtype=int)
    U = matmul(S, kron(C, I))

    if t == 1:
        return U
    else:
        U_t = U
        for i in range(t):
            U_t = matmul(U_t,U)
        return U_t

def get_quantum_walk_state(time, init_state):

    init = init_state

    t = time
    U = unitary_operator(t)
    qw_state = matmul(U, init)

    return qw_state


def get_prob_i(time, init_state):

    init = init_state

    t = time
    pi_all = []

    qw_state = get_quantum_walk_state(t, init)

    for coeff in qw_state:

        prob_tmp = np.around(coeff**2, decimals = 3)
        pi_all.append(prob_tmp)

    m = len(pi_all)
    P_i = []
    for k in range(0, m, 5):
        temp = []
        for l in range(N):
            temp.append(pi_all[l+k])
        P_i.append(temp)

    P_temp = P_i[0]
    for p in range(1, len(P_i)):
        P_temp = add(P_temp, P_i[p])

    return P_temp


def get_transition_matrix(time):

    P = []
    t = time

    for i in range(N):
        # print(i)
        init_qw = kron(Cs[0], Ps[i])

        P_i = get_prob_i(t, init_qw)

        P.append(P_i)

    P = vec(P)

    return P


def main(time):

    t = time
    P = get_transition_matrix(t)

    for pi in P:
        print('sum of p_i: ', sum(pi))

    print('transition matrix P:')
    print(P)

    vectices = ["1", "2", "3", "4","5"]

    fig, ax = plt.subplots()

    im, cbar = ht.heatmap(P, vectices, vectices, ax=ax,cmap="YlGn", cbarlabel="probability")
    texts = ht.annotate_heatmap(im, valfmt="{x:.3f}")

    # plt.title('P, %d' %t)

    fig.tight_layout()

    plt.show()

if __name__ == '__main__':

    t = 1000
    main(t)
