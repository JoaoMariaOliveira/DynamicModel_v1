import numpy as np

# ============================================================================================
# Labor Function that calculates
# ============================================================================================
def Labor(Y, N, J, D, T, beta, migracao, L):
    Ybeta = Y ** beta
    Migr0 = np.ones([(T+1) * D, D],dtype=float)

    for i in range(D):
        for d in range(D):
            Migr0[i, d] = migracao[i, d]

    AuxMigrSum = 0
    AuxMigr = np.ones([(T+1) * D, D], dtype=float)
    for t in range(T):
        for i in range((t+1)*D,(t+2)*D,1):
            for d in range(D):
                AuxMigr[i, d] = Migr0[i-D, d] * Ybeta[t, d]
        AuxMigrSum = sum(AuxMigr.transpose())
        for i in range((t+1)*D,(t+2)*D,1):
            for d in range(D):
                Migr0[i, d] = Migr0[i-D, d] * Ybeta[t, d] / AuxMigrSum [i]

    del AuxMigr, AuxMigrSum
    Distr0   = np.vstack((L,np.ones([T, D], dtype=float)))
    DistrAux = np.ones([D,1], dtype=float)
    AuxMigr  = np.ones([D,1], dtype=float)
    for t in range(T):
        for d in range(D):
            for i in range(D):
                DistrAux[i,0] = Distr0[t,i]
                AuxMigr[i,0]  = Migr0[(t+1)*D+i,d]
            Distr0[t+1,d] = sum(DistrAux*AuxMigr)

    CrescTrab = np.ones([T, D], dtype=float)
    for t in range(T):
        for d in range(D):
            CrescTrab[t,d] = Distr0[t+1,d] / Distr0[t,d]

    return CrescTrab, Migr0
