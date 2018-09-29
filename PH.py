import numpy as np


def PH(wages_N, tau_hat, mLinearThetas, mThetas, B, G, Din, nSectors, nCountries, nMaxIteracions, nTolerance, wbr0,
       nPositionBR, mPriceFactor, LG, Csi):

    # Adjusting Tolerance
    nToleranceLocal = 1E-07
    # reformatting theta vector
#    mLinearThetas = np.ones([nSectors*nCountries,1], dtype=float)
#    for j in range(nSectors):
#        for n in range(nCountries):
#            mLinearThetas[j * nCountries + n, 0] = mThetas[j]

    # initialize vectors of ex-post wage and price factors
    wf0 = np.ndarray.copy(wages_N)

    nMaxDiffPrice = 1
    nIteraction = 1

    mLogCost = np.ones([nSectors, nCountries], dtype=float)
    mCost = np.ones([nSectors, nCountries], dtype=float)
    Din_om = np.zeros([nSectors * nCountries, nCountries], dtype=float)

    while nIteraction <= nMaxIteracions and nMaxDiffPrice > nTolerance:
        lw = np.log(wf0)
        mLogPrice = np.log(mPriceFactor)
        lwbr = np.log(wbr0)

        # calculating log cost
        for i in range(nCountries):
            mLogCost[:, i] = ((B[:, i]*lw[i]).reshape(nSectors, 1) + ((1 - B[:, i]).reshape(nSectors, 1) *
                            np.dot(G[i * nSectors: (i + 1) * nSectors, :].T, mLogPrice[:, i].reshape(nSectors, 1))).reshape(nSectors, 1)).reshape(nSectors)

        for j in range(nSectors):
            mLogCost[j,nPositionBR] = (B[j, nPositionBR] * lwbr[j]) + \
                                      (B[j, nPositionBR] * Csi[j, 0] * np.log(LG[j, 0])) + (1 - B[j, nPositionBR])\
                                      * np.dot(G[nPositionBR * nSectors:(nPositionBR + 1) * nSectors, j].T ,mLogPrice[:, nPositionBR])

        mCost = np.exp(mLogCost)

        Din_om = Din * tau_hat ** (-1/(mLinearThetas * np.ones([1, nCountries], dtype=float)))

        # calculating phat
        phat = np.zeros([nSectors, nCountries], dtype=float)
        for j in range(nSectors):
            for n in range(nCountries):
                phat[j, n] = Din_om[n + j * nCountries, :].dot(mCost[j, :] ** (-1 / mThetas[j])).T

                # this happens because of sectors with zero VA
                # Note that we do not want logs of zero
                if phat[j, n] == 0:
                    phat[j, n] = 1
                else:
                    phat[j, n] = phat[j, n] ** (-mThetas[j])

        # Checking tolerance
        mPriceDiff = abs(phat - mPriceFactor)
        mPriceFactor = phat
        nMaxDiffPrice = np.amax(mPriceDiff)
        nIteraction += 1

    return mPriceFactor, mCost
