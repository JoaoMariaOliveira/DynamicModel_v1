# ============================================================================================
# EquiliBrium_LC function
# ported to Python by  João Maria
# ============================================================================================
import numpy as np
from LMC import LMC
from Dinprime import Dinprime
from Expenditure import Expenditure
from PH import PH

@profile
def equilibrium_LC(mTauHat, mTauActual, mAlphas, mLinearThetas, mThetas, mShareVA, mIO, Din, nSectors, nCountries,
                   nMaxIteracions, nTolerance, VAn, Sn, vfactor, LG, VA_Br, nBeta, nPositionBR, nActualYear,
                   w_aux, wbr_aux, mCsiBrasil, Csi_teste, mIota):
    mWages = w_aux[:, nActualYear].reshape(nCountries, 1)
    mWagesBrasil = wbr_aux[:, nActualYear].reshape(nSectors, 1)
    mPriceFactor = np.ones([nSectors, nCountries], dtype=float)
    PQ = 1000 * np.ones([nCountries, nSectors], dtype=float)
    nInterations = 1
    wfmax = 1.0
    while (nInterations <=nMaxIteracions) and (wfmax > nTolerance):

        mPriceFactor, mCost = PH(mWages, mTauHat, mLinearThetas, mThetas, mShareVA, mIO, Din, nSectors, nCountries,
                                 nMaxIteracions, nTolerance, mWagesBrasil, nPositionBR, mPriceFactor, LG, mCsiBrasil)

        # Calculating trade shares
        mTradeShare = Dinprime(Din, mTauHat, mCost, mLinearThetas, mThetas, nSectors, nCountries)
        mTradeShareOM = mTradeShare / mTauActual
        idxs = np.arange(0, nCountries) + (np.arange(nSectors) * nCountries)[:,None]
        mWeightedTariffs = np.sum((mTradeShare[idxs,:]/mTauActual[idxs,:]).T, axis=0).T
        # mWeightedTariffs_old = np.zeros([nSectors, nCountries], dtype=float)
        # for j in range(nSectors):
        #     mWeightedTariffs_old[j, :] = sum((mTradeShare[j * nCountries: (j + 1) * nCountries: 1, :] / mTauActual[j * nCountries: (j + 1) * nCountries: 1, :]).T)
        # assert np.array_equal(mWeightedTariffs, mWeightedTarrifs_old)

        # Expenditure matrix
        PQ = Expenditure(mAlphas, mShareVA, mIO, mTradeShare, mTauActual, mWeightedTariffs, VAn, mWages, Sn, nSectors, nCountries,
                         LG, VA_Br, mWagesBrasil, nPositionBR, PQ)
        # Iterating using LMC
        mWagesAux, mWagesBrasilAux = LMC(PQ, mTradeShareOM, nSectors, nCountries, mShareVA, VAn, VA_Br, nPositionBR, LG)
        # Excess function
        # ZW = (mWagesAux.reshape(N,1) - mWages.reshape(N,1))
        # ZW_T = sum(abs(mWages - mWagesAux));
        # ZW_Br = sum(abs(mWagesBrasil - mWagesBrasilAux))
        ZW = (mWagesBrasilAux - mWagesBrasil)
        PQ_vec = PQ.T.reshape(nSectors * nCountries, 1, order='F').copy()
        DP = mTradeShareOM * PQ_vec
        # DP_old = np.zeros([nSectors * nCountries, nCountries], dtype=float)
        # for n in range(nCountries):
        #     DP_old[:, n] = mTradeShareOM[:, n] * PQ_vec[:, 0]
        # assert np.array_equal(DP, DP_old)

        # Exports
        LHS = sum(DP).reshape(nCountries, 1)
        # calculating RHS (Imports) trade balance
        PF = PQ * mWeightedTariffs
        # Imports
        RHS = sum(PF).reshape(nCountries, 1)
        # Sn_pos = -(RHS - LHS)
        xbilattau = (PQ_vec * np.ones([1, nCountries], dtype=float)) * mTradeShareOM
        idxs = np.arange(0, nCountries) + (np.arange(nSectors) * nCountries)[:,None]
        GO = np.sum(xbilattau[idxs,:], axis=1)
        # GO_old = np.ones([nSectors, nCountries], dtype=float)
        # for j in range(nSectors):
        #     GO_old[j, :] = sum(xbilattau[j * nCountries: (j + 1) * nCountries, :])
        # assert np.array_equal(GO, GO_old)

        VAjn_pos = GO * mShareVA
        Cap_pos = VAjn_pos * Csi_teste
        rem_cap_pos = sum(Cap_pos).reshape(nCountries, 1)
        Qui_pos = sum(rem_cap_pos)
        iota_pos = (rem_cap_pos - Sn) / Qui_pos
        ZW2 = iota_pos - mIota
        # Excess function (trade balance)
        Snp = (RHS - LHS) + Sn
        ZW2 = -(RHS - LHS + Sn) / VAn
        # Itaration factor prices
        mWagesAux = mWages * (1 - vfactor * ZW2 / mWages)
        mWagesBrasilTemp = mWagesBrasil * (1 - vfactor * ZW / mWagesBrasil)
        wfmax = sum(abs(ZW2))
        mWages = mWagesAux
        mWagesBrasil = mWagesBrasilTemp
        nInterations += 1

    return mWages, mPriceFactor, PQ, mWeightedTariffs, mTradeShare, ZW, Snp, mCost, DP, PF, mWagesBrasil
