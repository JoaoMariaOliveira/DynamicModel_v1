# ============================================================================================
# Eexpenditure Function
# Portada para Python por João Maria em 26/4/2018
# ============================================================================================
""" This function calculates de Omega matrix in Caliendo - Parro (2009)
 Inputs are A = alphas, B = bethas, G = I-O matrix, Dinp = trade shares,
 tarifap = tarifs, mWeightedTariffs = trade weighted tariffs """

def kron(a, nrows):
    return np.repeat(a, nrows * np.ones(a.shape[0], np.int), axis=0)

import numpy as np
from cfuncs import ExpenditureAux, OM_sum_f, compute_IA, compute_NBP

def OM_sum(GG, NNBP, IA, nSectors, nCountries):
    GP = GG * NNBP
    I = np.eye(nSectors * nCountries, nSectors * nCountries, dtype=float)
    OM = I - (GP + IA)
    return OM

# def compute_IA(mWeightedTariffs, mAlphas, nSectors, nCountries):
#     IA = np.zeros([nSectors * nCountries, nSectors * nCountries], dtype=float)
#     I_F = 1 - mWeightedTariffs

#     for n in range(nCountries):
#         IA[n * nSectors: (n + 1) * nSectors, n * nSectors: (n + 1) * nSectors] =  np.kron(mAlphas[:, n], (I_F[:, n].T)).reshape(nSectors, nSectors)
#     return IA

# def compute_NBP(mTradeShare, mTauActual, mShareVA, nSectors, nCountries):
#     Pit = mTradeShare / mTauActual
#     Bt = 1 - mShareVA
#     BP = np.zeros([nSectors * nCountries, nCountries], dtype=float)

#     for j in range(nSectors):
#         BP[j * nCountries: (j + 1) * nCountries, :] = np.kron(np.ones(nCountries).reshape(nCountries, 1), Bt[j, :]) * Pit[j * nCountries: (j + 1) * nCountries, :]

#     NBP = np.zeros([nCountries, nSectors * nCountries], dtype=float)

#     for j in range(nCountries):
#         for n in range(nCountries):
#             NBP[j, n * nSectors: (n + 1) * nSectors] = BP[n: nSectors * nCountries: nCountries, j]
#     return NBP


@profile
def Expenditure(mAlphas, mShareVA, mIO, mTradeShare, mTauActual, mWeightedTariffs, VAn, mWages, Sn, nSectors, nCountries,
                LG, VA_Br, mWagesBrasil, nPositionBR, PQ):

    IA = compute_IA(mWeightedTariffs, mAlphas, nSectors, nCountries)

    Pit = mTradeShare / mTauActual
    Bt = 1 - mShareVA
    NBP = compute_NBP(Pit, Bt, nSectors, nCountries)

    NNBP = kron(NBP, nSectors)
    # NNBP_old = np.kron(NBP, np.ones([nSectors, 1], dtype=float))
    # assert np.array_equal(NNBP, NNBP_old)
    GG = np.tile(mIO, nCountries)
    # GG_old = np.kron(np.ones([1, nCountries], dtype=float), mIO)
    # assert np.array_equal(GG, GG_old)
    OM = OM_sum_f(GG, NNBP, IA, nSectors, nCountries)
    # OM_old = OM_sum(GG, NNBP, IA, nSectors, nCountries)
    # assert np.array_equal(OM, OM_old)

    A = np.kron(np.ones([nSectors, 1], dtype=float), (mWages * VAn).T) #.reshape(1, N))
    mShareVA = mWagesBrasil * LG * VA_Br
    C = np.sum(mShareVA)
    A[:, nPositionBR] = C

    Vb = mAlphas * A

    Vb = Vb.reshape(nSectors * nCountries, 1, order='F')
    Bb = -mAlphas * (Sn * np.ones((1, nSectors))).T
    Bb = Bb.reshape(nSectors * nCountries, 1, order='F')
    PQ_vec = PQ.T.reshape(nSectors * nCountries, 1, order='F')

    Soma = Vb + Bb
    PQ = ExpenditureAux(OM, Soma, PQ_vec)

    #temp = matrix_power(OM, -1)
    #DD1 = temp.dot(Vb)
    #DD2 = temp.dot(Bb)
    #PQ = DD1 + DD2
    return PQ.reshape(nSectors, nCountries, order='F')
