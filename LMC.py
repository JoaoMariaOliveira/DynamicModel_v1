"""
Calculating new wages using the labor market clearing condition

"""
import numpy as np


def LMC(Xp, Dinp, J, N, B, VAL, VA_Br, Pos, LG):
    PQ_vec = Xp.T.reshape(J * N, 1, order='F').copy()

    # Check if DDinpt gives a different value
    DDinpt = np.zeros((J * N, N))
    for n in range(N):
        DDinpt[:, n] = Dinp[:, n] * PQ_vec [:, 0]

    DDDinpt = np.zeros((J, N))
    for n in range(J):
        DDDinpt[n, :] = sum(DDinpt[n * N: (n + 1) * N, :])

    aux4 = B * DDDinpt
    aux5 = sum(aux4).reshape(N,1)
    wf0 = (1 / VAL) * aux5

    aux6 = aux4[:, Pos].reshape(J, 1)
    w_Br = aux6 / VA_Br

    w_Br = w_Br / LG
    #   w_Br = w_Br.T

    return wf0, w_Br
