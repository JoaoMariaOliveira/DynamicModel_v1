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

def ExpenditureAux(double[:, :] OM, double[:, :] Soma, np.ndarray[DTYPE_t, ndim=2] PQ):
    cdef np.ndarray[DTYPE_t, ndim=2] Dif
    cdef double PQmax = 1.
    while PQmax > 1E-03:
        Dif = np.dot(OM, PQ) - Soma
        PQmax = absmax(Dif)
        PQ = PQ - Dif
    return PQ


@cython.boundscheck(False)
@cython.wraparound(False)
def PHf(double[:,::1] Din_om, double[:,::1] mCost, np.ndarray[DTYPE_t, ndim=1] mThetas, Py_ssize_t nSectors, Py_ssize_t nCountries):
    cdef np.ndarray[DTYPE_t, ndim=2] mPriceHat = np.zeros([nSectors, nCountries], dtype=float)
    cdef double[:,:] mPriceHatView = mPriceHat
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
