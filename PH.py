import numpy as np
from cfuncs import PH_F

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


def PH(mWages, mTauHat, mLinearThetas, mThetas, mShareVA, G, Din, nSectors, nCountries, nMaxIterations, nTolerance,
       mWagesBrasil, nPositionBR, mPriceFactor, LG, mCsiBrasil):
    return PH_F(mWages, mTauHat, mLinearThetas, mThetas, mShareVA, G, Din, nSectors, nCountries, nMaxIterations, nTolerance,
        mWagesBrasil, nPositionBR, mPriceFactor, LG, mCsiBrasil)
