import numpy as np

def Dinprime(Din, mTauHat, mCost, mLinearThetas, mThetas, nSectors, nCountries):
    # reformatting theta vector
#    mLinearThetas = np.ones([nSectors * nCountries,1], dtype=float)
#    for j in range(nSectors):
#        for n in range(nCountries):
#            mLinearThetas[j * nCountries + n, :] = mThetas[j]

    pwr = (-1 / mThetas.reshape(1, nSectors))
    cp = mCost ** pwr.T
    # cp_old = np.ones(mCost.shape)
    # for n in range(nCountries):
    #     cp_old[:, n] = mCost[:, n] ** pwr
    # assert np.array_equal(cp, cp_old)

    Din_om = Din * (mTauHat ** (-1 / (mLinearThetas * np.ones([1, nCountries]))))

    idx = np.arange(0, nSectors*nCountries, nCountries) + np.arange(nCountries)[:,None]
    DD = (Din_om[idx] * cp).reshape(-1, nCountries, order='F')
    # DD_old = np.zeros([nSectors * nCountries, nCountries], dtype=float)
    # for n in range(nCountries):
    #     idx = np.arange( n, nSectors * nCountries - (nCountries - (n + 1) ), nCountries)
    #     DD_old[idx, :] = Din_om[idx, :] * cp
    # assert np.array_equal(DD, DD_old)

    phat = np.power(DD.sum(axis=1).T.reshape(nCountries * nSectors, 1), -mLinearThetas)
    Dinp = DD * (phat ** (1 / mLinearThetas))
    return Dinp
