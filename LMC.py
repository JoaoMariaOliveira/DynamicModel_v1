# ============================================================================================
# EquiliBrium Function
# Portada para Python por João Maria em 26/4/2018
# ============================================================================================"
#
#Calculating new wages using the labor market clearing condition
#
import numpy as np

def LMC(Xp, mTradeShareOM, nSectors, nCountries, mShareVA, VAL, VA_Br, nPositionBR, LG):
    PQ_vec = Xp.T.reshape(nSectors * nCountries, 1, order='F').copy()
    # Check if mTradeShareAux gives a different value
    mTradeShareAux = np.zeros((nSectors * nCountries, nCountries))
    for n in range(nCountries):
        mTradeShareAux[:, n] = mTradeShareOM[:, n] * PQ_vec [:, 0]

    mTradeShareTemp = np.zeros((nSectors, nCountries))
    for n in range(nSectors):
            mTradeShareTemp[n, :] = sum(mTradeShareAux[n * nCountries: (n + 1) * nCountries, :])

    mAux4 = mShareVA * mTradeShareTemp
    mAux5 = sum(mAux4).reshape(nCountries, 1)
    mWagesAux = (1 / VAL) * mAux5
    mAux6 = mAux4[:, nPositionBR].reshape(nSectors, 1)
    mWagesBrasilAux = mAux6 / VA_Br
    mWagesBrasilAux = mWagesBrasilAux / LG
    #   mWagesBrasilAux = mWagesBrasilAux.T

    return mWagesAux, mWagesBrasilAux