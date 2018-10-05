import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport fabs

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double absmax(double[:,::1] mat):
    cdef Py_ssize_t x_max = mat.shape[0]
    cdef Py_ssize_t y_max = mat.shape[1]
    cdef double mx = 0
    cdef double tmp;
    cdef Py_ssize_t x, y
    for x in range(x_max):
        for y in range(y_max):
            tmp = fabs(mat[x,y])
            if tmp > mx: mx = tmp
    return mx


@cython.boundscheck(False)
@cython.wraparound(False)
def ExpenditureAux(double[:, ::1] OM, double[:, ::1] Soma, np.ndarray[DTYPE_t, ndim=2] PQ):
    cdef np.ndarray[DTYPE_t, ndim=2] Dif
    cdef double PQmax = 1.
    cdef double tmp, tmp_mx;
    cdef Py_ssize_t x_max = PQ.shape[0]
    cdef Py_ssize_t x
    while PQmax > 1E-03:
        tmp_mx = 0
        Dif = np.dot(OM, PQ)
        for x in range(x_max):
            tmp = Dif[x, 0] - Soma[x, 0]
            PQ[x, 0] -= tmp
            tmp = fabs(tmp)
            if tmp > tmp_mx: tmp_mx = tmp
        PQmax = tmp_mx
    return PQ


@cython.boundscheck(False)
@cython.wraparound(False)
def PH_F(mWages, mTauHat, mLinearThetas, mThetas, mShareVA, G, Din, Py_ssize_t nSectors, Py_ssize_t nCountries, unsigned int nMaxIterations, double nTolerance,
       mWagesBrasil, Py_ssize_t nPositionBR, mPriceFactor, LG, mCsiBrasil):
    # initialize vectors of ex-post wage and price factors

    cdef Py_ssize_t i
    cdef unsigned int nMaxDiffPrice = 1
    cdef unsigned int nIteration = 1
    cdef np.ndarray[DTYPE_t, ndim=2] mLogCost = np.ones([nSectors, nCountries], dtype=float)
    mCost = np.ones([nSectors, nCountries], dtype=float)
    Din_om = np.zeros([nSectors * nCountries, nCountries], dtype=float)
    while nIteration <= nMaxIterations and nMaxDiffPrice > nTolerance:
        mLogWages = np.log(mWages)
        mLogPrice = np.log(mPriceFactor)
        mLogWagesBrasil = np.log(mWagesBrasil)
        # calculating log cost
        for i in range(nCountries):
            mLogCost[:, i] = ((mShareVA[:, i] * mLogWages[i]).reshape(nSectors, 1) + ((1 - mShareVA[:, i]).reshape(nSectors, 1)
             * np.dot(G[i * nSectors: (i + 1) * nSectors, :].T, mLogPrice[:, i].reshape(nSectors, 1))).reshape(nSectors, 1)).reshape(nSectors)

        for i in range(nSectors):
            mLogCost[i, nPositionBR] = (mShareVA[i, nPositionBR] * mLogWagesBrasil[i]) + (mShareVA[i, nPositionBR] * mCsiBrasil[i, 0]
             * np.log(LG[i, 0])) + (1 - mShareVA[i, nPositionBR]) * np.dot(G[nPositionBR * nSectors:(nPositionBR + 1) * nSectors, i].T ,mLogPrice[:, nPositionBR])

        mCost = np.exp(mLogCost)
        Din_om = Din * mTauHat ** (-1/(mLinearThetas * np.ones([1, nCountries], dtype=float)))
        mPriceHat = PHf(Din_om, mCost, mThetas.T[0], nSectors, nCountries)

        # Checking tolerance
        mPriceDiff = abs(mPriceHat - mPriceFactor)
        mPriceFactor = mPriceHat
        nMaxDiffPrice = np.amax(mPriceDiff)
        nIteration += 1

    return mPriceFactor, mCost


@cython.boundscheck(False)
@cython.wraparound(False)
cdef PHf(double[:,::1] Din_om, double[:,::1] mCost, np.ndarray[DTYPE_t, ndim=1] mThetas, Py_ssize_t nSectors, Py_ssize_t nCountries):
    cdef np.ndarray[DTYPE_t, ndim=2] mPriceHat = np.zeros([nSectors, nCountries], dtype=float)
    cdef double[:,::1] mPriceHatView = mPriceHat
    cdef double[:] powers = -1/mThetas
    cdef Py_ssize_t j, n
    for j in range(nSectors):
        for n in range(nCountries):
            mPriceHatView[j, n] = np.dot(Din_om[n + j * nCountries, :], np.power(mCost[j, :], powers[j])).T
            # this happens because of sectors with zero VA
            # Note that we do not want logs of zero
            if mPriceHatView[j, n] == 0:
                mPriceHatView[j, n] = 1
            else:
                mPriceHatView[j, n] = mPriceHatView[j, n] ** -mThetas[j]
    return mPriceHat


@cython.boundscheck(False)
@cython.wraparound(False)
def OM_sum_f(double[:,::1] GG, double[:,::1] NNBP, double[:,::1] IA, Py_ssize_t nSectors, Py_ssize_t nCountries):
    cdef np.ndarray[DTYPE_t, ndim=2] I = np.eye(nSectors * nCountries, nSectors * nCountries, dtype=float)
    cdef np.ndarray[DTYPE_t, ndim=2] OM = np.zeros([nSectors * nCountries, nSectors * nCountries], dtype=float)
    cdef double[:, ::1] I_view = I
    cdef double[:, ::1] OM_view = OM
    cdef Py_ssize_t i, j

    # These are all squares, all of the same dimensions
    # which is (nSectors*nCountries, nSectors*nCountries)
    cdef Py_ssize_t size = nSectors*nCountries
    for i in range(size):
        for j in range(size):
            OM_view[i, j] = I_view[i,j] - ((GG[i,j] * NNBP[i,j]) + IA[i,j])
    return OM

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_IA(mWeightedTariffs, mAlphas, Py_ssize_t nSectors, Py_ssize_t nCountries):
    cdef np.ndarray[DTYPE_t, ndim=2] IA = np.zeros([nSectors * nCountries, nSectors * nCountries], dtype=float)
    cdef Py_ssize_t n
    I_F = 1 - mWeightedTariffs

    for n in range(nCountries):
        IA[n * nSectors : (n + 1) * nSectors, n * nSectors: (n + 1) * nSectors] =  np.kron(mAlphas[:, n], (I_F[:, n].T)).reshape(nSectors, nSectors)
    return IA

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_NBP(Pit, Bt, Py_ssize_t nSectors, Py_ssize_t nCountries):
    cdef np.ndarray[DTYPE_t, ndim=2] BP = np.zeros([nSectors * nCountries, nCountries], dtype=float)
    cdef np.ndarray[DTYPE_t, ndim=2] NBP = np.zeros([nCountries, nSectors * nCountries], dtype=float)
    cdef Py_ssize_t j

    for j in range(nSectors):
        BP[j * nCountries: (j + 1) * nCountries, :] = np.kron(np.ones(nCountries).reshape(nCountries, 1), Bt[j, :]) * Pit[j * nCountries: (j + 1) * nCountries, :]

    for j in range(nCountries):
        for n in range(nCountries):
            NBP[j, n * nSectors: (n + 1) * nSectors] = BP[n: nSectors * nCountries: nCountries, j]
    return NBP

