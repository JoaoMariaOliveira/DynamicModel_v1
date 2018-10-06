# ============================================================================================
# EquiliBrium Function
# Portada para Python por João Maria em 26/4/2018
# ============================================================================================"
#
#Calculating new wages using the labor market clearing condition
#
import numpy as np

@profile
def LMC(Xp, mTradeShareOM, nSectors, nCountries, mShareVA, VAL, VA_Br, nPositionBR, LG):
    PQ_vec = Xp.T.reshape(nSectors * nCountries, 1, order='F').copy()
    # Check if mTradeShareAux gives a different value

    mTradeShareAux = mTradeShareOM * PQ_vec
    # mTradeShareAux_old = np.zeros((nSectors * nCountries, nCountries))
    # for n in range(nCountries):
    #     mTradeShareAux_old[:, n] = mTradeShareOM[:, n] * PQ_vec[:, 0]
    # assert np.array_equal(mTradeShareAux, mTradeShareAux_old)


    mTradeShareIdxs = np.arange(0, nCountries) + (np.arange(nSectors) * nCountries)[:,None]
    mTradeShareTemp = np.sum(mTradeShareAux[mTradeShareIdxs,:], axis=1)
    # mTradeShareTemp_old = np.zeros((nSectors, nCountries))
    # for n in range(nSectors):
    #     mTradeShareTemp_old[n, :] = sum(mTradeShareAux[n * nCountries: (n + 1) * nCountries, :])
    # assert np.array_equal(mTradeShareTemp, mTradeShareTemp_old)

    mAux4 = mShareVA * mTradeShareTemp
    mAux5 = sum(mAux4).reshape(nCountries, 1)
    mWagesAux = (1 / VAL) * mAux5
    mAux6 = mAux4[:, nPositionBR].reshape(nSectors, 1)
    mWagesBrasilAux = mAux6 / VA_Br
    mWagesBrasilAux = mWagesBrasilAux / LG
    #   mWagesBrasilAux = mWagesBrasilAux.T

    return mWagesAux, mWagesBrasilAux