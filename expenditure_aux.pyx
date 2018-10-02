import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_t

# BEST, 316.851
def ExpenditureAux(double[:, :] OM, double[:, :] Soma, np.ndarray[DTYPE_t, ndim=2] PQ):
    cdef np.ndarray[DTYPE_t, ndim=2] Dif
    cdef double PQmax = 1.
    while PQmax > 1E-03:
        Dif = np.dot(OM, PQ) - Soma
        PQmax = np.max(np.abs(Dif))
        PQ = PQ - Dif
    return PQ
