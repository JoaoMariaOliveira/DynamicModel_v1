""" This function calculates de Omega matrix in Caliendo - Parro (2009)
 Inputs are A = alphas, B = bethas, G = I-O matrix, Dinp = trade shares,
 tarifap = tarifs, Fp = trade weighted tariffs """

import numpy as np



def ExpenditureAux(OM, Vb, Bb, N, J, PQ):
    Soma = Vb + Bb
    PQmax = 1
    while PQmax > 1E-07:
        Dif = np.dot(OM, PQ) - Soma
     #   Dif_aux = abs(Dif)
        PQmax = max(abs(Dif))
        PQ = PQ - Dif

    return PQ


def Expenditure(alphas, B, G, Dinp, taup, Fp, VAn, wf0, Sn, J, N, LG, VA_Br, wbr0, Pos, PQ):

    IA = np.zeros([J * N, J * N], dtype=float)
    I_F = 1 - Fp

    for n in range(N):
        IA[n * J: (n + 1) * J, n * J: (n + 1) * J] =  np.kron(alphas[:, n], (I_F[:, n].T)).reshape(J, J)

    Pit = Dinp/taup
    Bt = 1 - B
    BP = np.zeros([J*N,N], dtype=float)

    for j in range(J):
        BP[j * N: (j + 1) * N, :] = np.kron(np.ones(N).reshape(N, 1), Bt[j, :]) * Pit[j * N: (j + 1) * N, :]

    NBP = np.zeros([N, J*N], dtype=float)

    for j in range(N):
        for n in range(N):
            NBP[j, n * J: (n + 1) * J] = BP[n: J * N: N, j]

    NNBP = np.kron(NBP, np.ones([J, 1], dtype=float))
    GG = np.kron(np.ones([1, N], dtype=float), G)
    GP = GG * NNBP

    OM = np.eye(J * N, J * N, dtype=float) - (GP + IA)

    A = np.kron(np.ones([J, 1], dtype=float), (wf0 * VAn).T) #.reshape(1, N))
    B = wbr0 * LG * VA_Br
    C = np.sum(B)
    A[:, Pos] = C

    Vb = alphas * A

    Vb = Vb.reshape(J * N, 1, order='F').copy()
    Bb = -alphas * (Sn * np.ones((1, J))).T
    Bb = Bb.reshape(J * N, 1, order='F').copy()
    PQ_vec = PQ.T.reshape(J * N, 1, order='F').copy()
    PQ = ExpenditureAux(OM, Vb, Bb, N, J, PQ_vec)


    #temp = matrix_power(OM, -1)
    #DD1 = temp.dot(Vb)
    #DD2 = temp.dot(Bb)
    #PQ = DD1 + DD2
    PQ = PQ.reshape(J, N, order='F').copy()

    return PQ
