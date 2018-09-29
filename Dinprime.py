import numpy as np

def Dinprime(Din, tau_hat, c, mLinearThetas, mThetas, nSectors, nCountries):

    # reformatting theta vector
#    mLinearThetas = np.ones([nSectors * nCountries,1], dtype=float)
#    for j in range(nSectors):
#        for n in range(nCountries):
#            mLinearThetas[j * nCountries + n, :] = mThetas[j]

    cp = np.ones(c.shape)
    for n in range(nCountries):
        cp[:, n] = c[:, n] ** (-1 / mThetas.reshape(1, nSectors))

    Din_om = Din * (tau_hat ** (-1 / (mLinearThetas * np.ones([1, nCountries]))))

    DD = np.zeros([nSectors * nCountries, nCountries], dtype=float)

    for n in range(nCountries):
        idx= np.arange( n, nSectors * nCountries - (nCountries - (n + 1) ), nCountries)
        DD[idx, :] = Din_om[idx, :] * cp

    phat = np.power(DD.sum(axis=1).T.reshape(nCountries * nSectors, 1), -mLinearThetas)

    Dinp = DD * (phat ** (1 / mLinearThetas))

    return Dinp
