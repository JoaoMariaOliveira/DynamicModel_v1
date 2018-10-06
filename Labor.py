import numpy as np

# ============================================================================================
# Labor Function that calculates
# ============================================================================================
@profile
def Labor(Y, N, J, D, T, beta, migracao, L):
    Ybeta = Y ** beta
    Migr0 = np.ones([(T+1) * D, D],dtype=float)
    Migr0[:D, :D] = migracao

    AuxMigrSum = 0
    AuxMigr = np.ones([(T+1) * D, D], dtype=float)

    # Migr0_old = Migr0.copy()
    # AuxMigr_old = AuxMigr.copy()
    # for t in range(T):
    #     for i in range((t+1)*D,(t+2)*D,1):
    #         for d in range(D):
    #             AuxMigr_old[i, d] = Migr0_old[i-D, d] * Ybeta[t, d]
    #     AuxMigrSum = sum(AuxMigr_old.transpose())
    #     for i in range((t+1)*D,(t+2)*D,1):
    #         for d in range(D):
    #             Migr0_old[i, d] = Migr0_old[i-D, d] * Ybeta[t, d] / AuxMigrSum[i]

    for t in range(T):
        idx = np.arange((t+1)*D,(t+2)*D)
        AuxMigr[idx, :] = Migr0[idx-D, :] * Ybeta[t, :]
        AuxMigrSum = sum(AuxMigr.transpose())
        for i in range((t+1)*D,(t+2)*D,1):
            Migr0[i, :] = Migr0[i-D, :] * Ybeta[t, :] / AuxMigrSum [i]

    # assert np.array_equal(Migr0, Migr0_old)
    # assert np.array_equal(AuxMigr, AuxMigr_old)

    del AuxMigr, AuxMigrSum
    Distr0   = np.vstack((L,np.ones([T, D], dtype=float)))
    DistrAux = np.ones([D,1], dtype=float)
    AuxMigr  = np.ones([D,1], dtype=float)
    idx = np.arange(D)
    for t in range(T):
        for d in range(D):
            DistrAux[:,0] = Distr0[t,:]
            AuxMigr[:,0]  = Migr0[(t+1)*D+idx,d]
            Distr0[t+1,d] = sum(DistrAux*AuxMigr)

    # Distr0_old   = np.vstack((L,np.ones([T, D], dtype=float)))
    # DistrAux_old = np.ones([D,1], dtype=float)
    # AuxMigr_old  = np.ones([D,1], dtype=float)
    # for t in range(T):
    #     for d in range(D):
    #         for i in range(D):
    #             DistrAux_old[i,0] = Distr0_old[t,i]
    #             AuxMigr_old[i,0]  = Migr0[(t+1)*D+i,d]
    #         Distr0_old[t+1,d] = sum(DistrAux_old*AuxMigr_old)
    # assert np.array_equal(Distr0, Distr0_old)
    # assert np.array_equal(AuxMigr, AuxMigr_old)
    # assert np.array_equal(DistrAux, DistrAux_old)


    CrescTrab = Distr0[1:]/Distr0[:-1]
    # CrescTrab_old = np.ones([T, D], dtype=float)
    # for t in range(T):
    #     for d in range(D):
    #         CrescTrab_old[t,d] = Distr0[t+1,d] / Distr0[t,d]
    # np.array_equal(CrescTrab, CrescTrab_old)

    return CrescTrab, Migr0
