import numpy as np
from cfuncs import PHf

def PH_subroutine(Din_om, mCost, mThetas, nSectors, nCountries):
    mPriceHat = np.zeros([nSectors, nCountries], dtype=float)
    for j in range(nSectors):
        for n in range(nCountries):
            mPriceHat[j, n] = Din_om[n + j * nCountries, :].dot(mCost[j, :] ** (-1 / mThetas[j])).T
            # this happens because of sectors with zero VA
            # Note that we do not want logs of zero
            if mPriceHat[j, n] == 0:
                mPriceHat[j, n] = 1
            else:
                mPriceHat[j, n] = mPriceHat[j, n] ** (-mThetas[j])
    return mPriceHat


@profile
def PH(mWages, mTauHat, mLinearThetas, mThetas, mShareVA, G, Din, nSectors, nCountries, nMaxIterations, nTolerance,
       mWagesBrasil, nPositionBR, mPriceFactor, LG, mCsiBrasil):

    #nToleranceLocal = 1E-07
    nToleranceLocal= nTolerance
    # initialize vectors of ex-post wage and price factors
    mWagesAux = np.ndarray.copy(mWages)

    nMaxDiffPrice = 1
    nIteration = 1
    mLogCost = np.ones([nSectors, nCountries], dtype=float)
    mCost = np.ones([nSectors, nCountries], dtype=float)
    Din_om = np.zeros([nSectors * nCountries, nCountries], dtype=float)
    while nIteration <= nMaxIterations and nMaxDiffPrice > nToleranceLocal:
        mLogWages = np.log(mWagesAux)
        mLogPrice = np.log(mPriceFactor)
        mLogWagesBrasil = np.log(mWagesBrasil)
        # calculating log cost
        for i in range(nCountries):
            mLogCost[:, i] = ((mShareVA[:, i] * mLogWages[i]).reshape(nSectors, 1) + ((1 - mShareVA[:, i]).reshape(nSectors, 1)
             * np.dot(G[i * nSectors: (i + 1) * nSectors, :].T, mLogPrice[:, i].reshape(nSectors, 1))).reshape(nSectors, 1)).reshape(nSectors)

        for j in range(nSectors):
            mLogCost[j,nPositionBR] = (mShareVA[j, nPositionBR] * mLogWagesBrasil[j]) + (mShareVA[j, nPositionBR] * mCsiBrasil[j, 0]
             * np.log(LG[j, 0])) + (1 - mShareVA[j, nPositionBR]) * np.dot(G[nPositionBR * nSectors:(nPositionBR + 1) * nSectors, j].T ,mLogPrice[:, nPositionBR])

        mCost = np.exp(mLogCost)
        Din_om = Din * mTauHat ** (-1/(mLinearThetas * np.ones([1, nCountries], dtype=float)))
        # calculating mPriceHat
        # mPriceHat = PH_subroutine(Din_om, mCost, mThetas, nSectors, nCountries)
        mPriceHat = PHf(Din_om, mCost, mThetas.T[0], nSectors, nCountries)

        # Checking tolerance
        mPriceDiff = abs(mPriceHat - mPriceFactor)
        mPriceFactor = mPriceHat
        nMaxDiffPrice = np.amax(mPriceDiff)
        nIteration += 1

    return mPriceFactor, mCost
