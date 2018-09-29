import numpy as np
from LaborFunctions import Labor
import SupportFunctions as FuncoesApoio
from LC import equilibrium_LC
import time
# ============================================================================================
# Função EquiliBrium
# Portada para Python por João Maria em 26/4/2018
# ============================================================================================


def Equilibrium(nCountries, nSectors, TS, D, nYears, beta, v, Pos, migracao, L, vfactor, nMaxIteracions, nTolerance, Yinic, nAdjust, Csi, nChoque, sDirectoryInput, sDirectoryOutput):

    Y = Yinic

    # Loading trade flows in files txt
    # B         - Share of value added
    # GO        - Gross Output
    # T (Thetas)- dispersion of productivity - non-tradables = 8.22
    #               Need Checks that dispersion of productivity
    lRead = ['B', 'Comercio', 'Csi_total', "GO", 'IO', 'Tarifas', 'TarifasZero', 'T']
    B, Comercio, Csi_total, mGrossOutputOrigin, IO,Tarifas, TarifasZero, mThetasOrigin\
            = FuncoesApoio.read_data_txt(lRead, sDirectoryInput)
# ============================================================================================
# Loading data from prior run from csv files
    if nChoque == 0:
        lRead = ['w_aux', 'wbr_aux']
    else:
        lRead = ['w_aux_C', 'wbr_aux_C']
    w_aux, wbr_aux = FuncoesApoio.read_data_csv(lRead, sDirectoryOutput)

    nIteracion = 1
    Ymax = 1
    while (nIteracion <= nMaxIteracions) and (Ymax > nTolerance):

        CrescTrab, migr = Labor(Y, nCountries, nSectors, D, nYears, beta, migracao, L)

        mGrossOutput = np.copy(mGrossOutputOrigin)

        xbilat1993 = Comercio

        xbilat1993_new = np.vstack((xbilat1993, np.zeros([(nSectors - TS) * nCountries, nCountries])))

        # Reading Tariffs
        tau1993 = Tarifas


        if nChoque == 0:
            tau2005 = Tarifas

            print("+++++++++++++++++++++++++++++++")
            print("Running normal scenario")
            print("iteration ", nIteracion)
            print("+++++++++++++++++++++++++++++++")
        else:
            tau2005 = TarifasZero

            print("+++++++++++++++++++++++++++++++")
            print("Running counterfactual scenario")
            print("iteration ", nIteracion)
            print("+++++++++++++++++++++++++++++++")

        tau = np.vstack((1 + tau1993 / 100, np.ones([(nSectors - TS) * nCountries, nCountries], dtype=float)))  # actual tariff vector
        taup = np.vstack((1 + tau2005 / 100, np.ones([(nSectors - TS)* nCountries, nCountries], dtype=float)))  # counterfactual tariff vector

        # Reading parameters
        # IO Coefficients
        G = IO
        #
        mThetas = mThetasOrigin
        #
        # dispersion of productivity - non-tradables = 8.22
        # Need Checks that dispersion of productivity
        #
        mThetas = np.hstack((1. / mThetas, np.ones([(nSectors - TS)], dtype=float) * 1 / 8.22)).reshape(nSectors, 1)
        # reformatting theta vector
        mLinearThetas = np.ones([nSectors * nCountries, 1], dtype=float)
        for j in range(nSectors):
            for n in range(nCountries):
                mLinearThetas[j * nCountries + n, :] = mThetas[j]

        # Calculating expenditures
        xbilat = xbilat1993_new * tau

        # Domestic sales
        x = np.zeros([nSectors, nCountries])
        xbilat_domestic = xbilat / tau

        for i in range(nSectors):
            # Computing sum of partial columns (0 a 30, 31 sectors) of exports
            # Adding as rows
            x[i, :] = sum(xbilat_domestic[i * nCountries: (i + 1) * nCountries, :])

        # Checking MAX between Exports and Domestic Product
        mGrossOutput = np.maximum(mGrossOutput, x)
        domsales = mGrossOutput - x

        # Bilateral trade matrix
        domsales_aux = domsales.T
        aux2 = np.zeros([nSectors * nCountries, nCountries], dtype=float)
        for i in range(nSectors):
            aux2[i * nCountries: ((i + 1) * nCountries), :] = np.diag(domsales_aux[:, i])

        xbilat = aux2 + xbilat

        # Calculating Expenditures shares
        A = sum(xbilat.T)
        XO = np.zeros([nSectors, nCountries])

        for j in range(nSectors):
            XO[j, :] = A[j * nCountries: (j + 1) * nCountries]

        # Calculating expenditures shares
        Xjn = sum(xbilat.T).T.reshape(nSectors * nCountries, 1).dot(np.ones([1, nCountries], dtype=float))
        Din = xbilat / Xjn

        # Calculating superavits
        xbilattau = xbilat / tau
        M = np.zeros([nSectors, nCountries])
        E = np.zeros([nSectors, nCountries])
        for j in range(nSectors):
            # Imports
            M[j, :] = sum(xbilattau[j * nCountries: (j + 1) * nCountries, :].T).T
            for n in range(nCountries):
                # Exports
                E[j, n] = sum(xbilattau[j * nCountries: (j + 1) * nCountries, n]).T

        Sn = (sum(E).T - sum(M).T).reshape(nCountries, 1)

        # Calculating Value Added
        VAjn = mGrossOutput * B
        VAn = sum(VAjn).T.reshape(nCountries, 1)

        VA_Br = np.ones([nSectors, 1], dtype= float)
        for j in range(nSectors):
            VA_Br[j, 0] = VAjn[j, Pos]
## olhar onde o CAP é
        Csi_teste = Csi_total
        Cap = VAjn * Csi_teste
        rem_cap = sum(Cap).T.reshape(nCountries, 1)
        Qui = sum(rem_cap)
        iota = (rem_cap - Sn) / Qui

        num = np.zeros([nSectors, nCountries])
        for n in range(nCountries):
            num[:, n] = XO[:, n] - G[n * nSectors:(n + 1) * nSectors, :].dot((1 - B[:, n]) * E[:, n])

        F = np.zeros([nSectors, nCountries])
        for j in range(nSectors):
            F[j, :] = sum((Din[j * nCountries: (j + 1) * nCountries:1, :] / tau[j * nCountries: (j + 1) * nCountries:1, :]).T)

        alphas = num / (np.ones([nSectors, 1], dtype=float)).dot((VAn + sum(XO * (1 - F)).T.reshape(nCountries, 1) - Sn).T)

        for j in range(nSectors):
            for n in range(nCountries):
                if alphas[j, n] < 0:
                    alphas[j, n] = 0

        alphas = alphas / np.ones([nSectors, 1]).dot(sum(alphas).reshape(1, nCountries))

        ##############################
        # Main program conterfactuals
        ##############################

        VAn = VAn / 100
        Sn = Sn / 100
        VA_Br = VA_Br / 100

        VABrasil = np.ones([nYears, nSectors], dtype=float)
        w_Brasil = np.ones([nYears, nSectors], dtype=float)
        P_Brasil = np.ones([nYears, nSectors], dtype=float)
        PBr = np.ones([nYears, 1], dtype=float)
        xbilat_total = np.zeros([nYears * nSectors * nCountries, nCountries], dtype=float)
        mGrossOutputTotal = np.zeros([nYears * nSectors, nCountries], dtype=float)
        p_total = np.zeros([nYears * nSectors, nCountries], dtype=float)

        # ============================================================================================
        # Routine that repeat for nYears years
        # ============================================================================================
        for t in range(nYears):

            print("Running for t = ", t)
            TInicio = time.perf_counter()
            print("Begin: ", time.strftime("%d/%b/%Y - %H:%M:%S", time.localtime()))

            LG = np.ones([nSectors, 1], dtype=float)
            for j in range(nSectors):
                LG[j, 0] = CrescTrab[t, j + 1]
     #       s = t

            if t == 0:  # First Year
                tau_hat = taup / tau
            else:
                tau_hat = taup / taup

            wf0, pf0, PQ, Fp, Dinp, ZW, Snp2, c, DP, PF, w_Br \
                = equilibrium_LC(tau_hat, taup, alphas, mLinearThetas, mThetas, B, G, Din, nSectors, nCountries,
                                 nMaxIteracions, nTolerance, VAn, Sn, vfactor, LG, VA_Br, beta, Pos, t, w_aux, wbr_aux,
                                 Csi, Csi_teste, iota)

            w_aux = np.ones([nCountries, nYears], dtype=float)
            wbr_aux = np.ones([nSectors, nYears], dtype=float)
            for n in range(nCountries):
                w_aux[n, t] = wf0[n, 0]

            for j in range(nSectors):
                wbr_aux[j, t] = w_Br[j, 0]

            PQ_vec = PQ.T.reshape(nSectors * nCountries, 1, order='F').copy()  # expenditures Xji in long vector: PQ_vec=(X11 X12 X13...)'
            Dinp_om = Dinp / taup
            xbilattau = (PQ_vec.dot(np.ones((1, nCountries)))) * Dinp_om
            xbilatp = xbilattau * taup

            for j in range(nSectors):
                mGrossOutput[j, :] = sum(xbilattau[j * nCountries: (j + 1) * nCountries, :])

            VAjn = mGrossOutput * B
            VAn = sum(VAjn).T.reshape(nCountries, 1)
            # dif no VA_Br 2/5/2018 00:51
            VA_Br = VAjn[:, Pos].reshape(nSectors, 1)

            Din = Dinp

            for j in range(nSectors):
                VABrasil[t, j] = VA_Br[j, 0]
                w_Brasil[t, j] = w_Br[j, 0]
                P_Brasil[t, j] = pf0[j, Pos]

            # pf0_all = pf0. / (alphas);
            # P = prod(pf0_all. ^ (alphas));
            # PBr(t, 1) = P(1, Pos);

            # pf0_all = pf0./(alphas);
            P = np.prod(pf0 ** (alphas), axis=0)
            PBr[t, 0] = P[Pos]

            for j in range(nSectors):
                for n in range(nCountries):
                    xbilatp[n + j * nCountries, n] = 0

            for i in range(nCountries * nSectors):
                for n in range(nCountries):
                    xbilat_total[t * nCountries * nSectors + i, n] = xbilatp[i, n]

            for j in range(nSectors):
                for n in range(nCountries):
                    mGrossOutputTotal[t * nSectors + j, n] = mGrossOutput[j, n]

                    p_total[t * nSectors + j, n] = pf0[j, n]

            print("End    : ", time.strftime("%d/%b/%Y - %H:%M:%S", time.localtime()))
            TFim = time.perf_counter()
            TDecorrido = (TFim - TInicio)
            print("Spent: %.2f segs" % TDecorrido)

#        Y_aux = Y
        Y_aux = np.ones([nYears, D], dtype=float)

        for i in range(nYears - 1, 0, -1):
            Y_aux[i - 1, 0] = np.dot(migr[(i - 1) * D, :], (Y_aux[i, :] ** beta).T.reshape(D, 1))

            for j in range(nSectors):
                Y_aux[i - 1, j + 1] = np.dot(((w_Brasil[i - 1, j] / PBr[i - 1]) ** (1 / v)),
                                             np.dot(migr[(i - 1) * D + j + 1, :], (Y_aux[i, :] ** beta).T))

        Y1_ant = Y
        Y1 = Y_aux

        Y_aux2 = sum(abs(Y1 - Y))

        vYmax = np.zeros([1, 1], dtype=float)
        vYmax[0, 0] = sum(Y_aux2.T)
        Ymax = vYmax[0, 0]

        print( "Ymax = ", Ymax)

        Y2 = Y - nAdjust * (Y - Y1)

        if nChoque == 0:
            lDataToSave = ['Y1', 'w_aux', 'wbr_aux', 'Y1_ant', 'YmaxV']
        else:
            lDataToSave = ['Y1_C', 'w_aux_C', 'wbr_aux_C', 'Y1_ant_C', 'YmaxV_C']

        lData = [Y1, w_aux, wbr_aux, Y1_ant, vYmax]
        FuncoesApoio.write_data_csv(lDataToSave, lData, sDirectoryOutput)

        Y = Y2

        nIteracion +=1

    return VABrasil, w_Brasil, P_Brasil, Y, CrescTrab, PBr, xbilat_total, mGrossOutputTotal, p_total, migr
