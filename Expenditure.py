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

def Expenditure(mAlphas, mShareVA, mIO, mTradeShare, mTauActual, mWeightedTariffs, VAn, mWages, Sn, nSectors, nCountries,
                LG, VA_Br, mWagesBrasil, nPositionBR, PQ, tolerance):

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
    PQ = ExpenditureAux(OM, Soma, PQ_vec, tolerance)

    #temp = matrix_power(OM, -1)
    #DD1 = temp.dot(Vb)
    #DD2 = temp.dot(Bb)
    #PQ = DD1 + DD2
    return PQ.reshape(nSectors, nCountries, order='F')
