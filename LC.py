import numpy as np
from LMC import LMC
from Dinprime import Dinprime
from Expenditure import Expenditure
from PH import PH

# ============================================================================================
# Função EquiliBrium_LC
# Portada para Python por João Maria em 26/4/2018
# ============================================================================================

def equilibrium_LC(tau_hat, taup, alphas, mLinearThetas, mThetas, B, G, Din, J, N, nMaxIteracions, nTolerance,
                   VAn, Sn, vfactor, LG, VA_Br, beta, Pos, t, w_aux, wbr_aux, Csi, Csi_teste, iota):
    # initialize vectors of ex-post wage and price factors
    # wf0 = np.ones([N, 1])
    # pf0 = np.ones([J, N])
    # wfmax = np.array([1.0])

    wf0 = w_aux[:, t].reshape(N, 1)
    wbr0 = wbr_aux[:, t].reshape(J, 1)
    pf0 = np.ones([J, N], dtype=float)
    PQ = 1000 * np.ones([N, J], dtype=float)

    e = 1.0
    wfmax = 1

    while (e <=nMaxIteracions) and (wfmax > nTolerance):

        pf0, c = PH(wf0, tau_hat, mLinearThetas, mThetas, B, G, Din, J, N, nMaxIteracions, nTolerance,
                    wbr0, Pos, pf0, LG, Csi)

        # Calculating trade shares
        Dinp = Dinprime(Din, tau_hat, c, mLinearThetas, mThetas, J, N)
        Dinp_om = Dinp / taup

        Fp = np.zeros([J, N], dtype=float)
        for j in range(J):
            Fp[j, :] = sum((Dinp[j * N: (j + 1) * N: 1, :] / taup[j * N: (j + 1) * N: 1, :]).T)

        # Expenditure matrix

        PQ = Expenditure(alphas, B, G, Dinp, taup, Fp, VAn, wf0, Sn, J, N, LG, VA_Br, wbr0, Pos, PQ)

        # Iterating using LMC
        wf1, w_Br = LMC(PQ, Dinp_om, J, N, B, VAn, VA_Br, Pos, LG)

        # Excess function
        # ZW = (wf1.reshape(N,1) - wf0.reshape(N,1))
        # ZW_T = sum(abs(wf0 - wf1));
        # ZW_Br = sum(abs(wbr0 - w_Br))


        ZW = (w_Br - wbr0)
        PQ_vec = PQ.T.reshape(J * N, 1, order='F').copy()
        DP = np.zeros([J * N, N], dtype=float)
        for n in range(N):
            DP[:, n] = Dinp_om[:, n] * PQ_vec [:, 0]

        # Exports
        LHS = sum(DP).reshape(N, 1)

        # calculating RHS (Imports) trade balance
        PF = PQ * Fp
        # Imports
        RHS = sum(PF).reshape(N, 1)

        Sn_pos = -(RHS - LHS)

        xbilattau = (PQ_vec * np.ones([1, N], dtype=float)) * Dinp_om
        GO = np.ones([J, N], dtype=float)
        for j in range(J):
            GO[j, :] = sum(xbilattau[j * N: (j + 1) * N, :])

        VAjn_pos = GO * B
        Cap_pos = VAjn_pos * Csi_teste
        rem_cap_pos = sum(Cap_pos).reshape(N, 1)
        Qui_pos = sum(rem_cap_pos)
        iota_pos = (rem_cap_pos - Sn) / Qui_pos
        ZW2 = iota_pos - iota

        # Excess function (trade balance)

        Snp = (RHS - LHS) + Sn
        ZW2 = -(RHS - LHS + Sn) / VAn

        # Itaration factor prices
        wf1 = wf0 * (1 - vfactor * ZW2 / wf0)

        wbr1 = wbr0 * (1 - vfactor * ZW / wbr0)

     #   Excesso = sum(abs(wbr0 - wbr1))

     #   wfmax = sum(abs(wf1 - wf0))
        wfmax = sum(abs(ZW2))

        wf0 = wf1
        wbr0 = wbr1

        e += 1
    return wf0, pf0, PQ, Fp, Dinp, ZW, Snp, c, DP, PF, wbr0
