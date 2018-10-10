# ============================================================================================
# EquiliBrium Function
# ported to Python by João Maria at 26/4/2018
# ============================================================================================
import numpy as np
import SupportFunctions as Support
from LC import equilibrium_LC
from cfuncs import Labor
import time

def Equilibrium(nCountries, nSectors, nTradebleSectors, nSectorsLabor, nYears, nBeta, nValIntertemp, nPositionBR,
                mInitialMigration, mInitialLaborStock, vFactor, nMaxIterations, nTolerance, mInitialY, nAdjust,
                mCsiBrasil, nCenario, sDirectoryInput, sDirectoryOutput):

    mY = mInitialY
    # Loading trade flows in files txt
    # B         - Share of value added
    # GO        - Gross Output
    # IO        - Input Output Matrix
    # T (Thetas)- dispersion of productivity - non-tradables = 8.22
    #               Need Checks that dispersion of productivity
    lRead = ['B', 'Comercio', 'Csi_total', "GO", 'IO', 'Tarifas', 'TarifasZero', 'T']
    mShareVA, mTrade, Csi_total, mGrossOutputOrigin, mIO, mNormalTariffs, mShockTariffs, mThetasOrigin\
            = Support.read_data_txt(lRead, sDirectoryInput)
# ============================================================================================
# Loading data from prior run from csv files
    if nCenario == 0:
        lRead = ['w_aux', 'wbr_aux']
    else:
        lRead = ['w_aux_C', 'wbr_aux_C']

    w_aux, wbr_aux = Support.read_data_csv(lRead, sDirectoryOutput)
    nIteration = 1
    Ymax = 1
    while (nIteration <= nMaxIterations) and (Ymax > nTolerance):
        mGrowthLabor, mMigration = \
            Labor(mY, nCountries, nSectors, nSectorsLabor, nYears, nBeta, mInitialMigration, mInitialLaborStock)

        mGrossOutput = np.copy(mGrossOutputOrigin)
        if nCenario == 0:
            mTauPrev = np.vstack((1 + mNormalTariffs / 100, np.ones([(nSectors - nTradebleSectors) * nCountries, nCountries],
                                                        dtype=float)))  # actual tariff vector
            mTauActual = np.vstack((1 + mNormalTariffs / 100, np.ones([(nSectors - nTradebleSectors) * nCountries, nCountries],
                                                         dtype=float)))  # counterfactual tariff vector
            print("+++++++++++++++++++++++++++++++")
            print("Running normal scenario")
            print("iteration ", nIteration)
            print("+++++++++++++++++++++++++++++++")
        else:
            mTauPrev = np.vstack((1 + mNormalTariffs / 100, np.ones([(nSectors - nTradebleSectors) * nCountries, nCountries],
                                                        dtype=float)))  # actual tariff vector
            mTauActual = np.vstack((1 + mShockTariffs / 100, np.ones([(nSectors - nTradebleSectors) * nCountries, nCountries],
                                                         dtype=float)))  # counterfactual tariff vector
            print("+++++++++++++++++++++++++++++++")
            print("Running counterfactual scenario")
            print("iteration ", nIteration)
            print("+++++++++++++++++++++++++++++++")

        mThetas = mThetasOrigin
        # dispersion of productivity - non-tradables = 8.22
        # Need Checks that dispersion of productivity
        mThetas = np.hstack((1. / mThetas, np.ones([(nSectors - nTradebleSectors)], dtype=float) * 1 / 8.22)).reshape(nSectors, 1)
        # reformatting theta vector
        mLinearThetas = np.ones([nSectors * nCountries, 1], dtype=float)
        for j in range(nSectors):
            for n in range(nCountries):
                mLinearThetas[j * nCountries + n, :] = mThetas[j]

        # Calculating expenditures
        xbilat = np.vstack((mTrade, np.zeros([(nSectors - nTradebleSectors) * nCountries, nCountries]))) * mTauPrev
        # Domestic sales
        x = np.zeros([nSectors, nCountries])
        xbilat_domestic = xbilat / mTauPrev
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
        xbilattau = xbilat / mTauPrev
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
        VAjn = mGrossOutput * mShareVA
        VAn = sum(VAjn).T.reshape(nCountries, 1)
        VA_Br = np.ones([nSectors, 1], dtype= float)
        for j in range(nSectors):
            VA_Br[j, 0] = VAjn[j, nPositionBR]

        Csi_teste = Csi_total
        Cap = VAjn * Csi_teste
        rem_cap = sum(Cap).T.reshape(nCountries, 1)
        Qui = sum(rem_cap)
        mIota = (rem_cap - Sn) / Qui
        num = np.zeros([nSectors, nCountries])
        for n in range(nCountries):
            num[:, n] = XO[:, n] - mIO[n * nSectors:(n + 1) * nSectors, :].dot((1 - mShareVA[:, n]) * E[:, n])

        F = np.zeros([nSectors, nCountries])
        for j in range(nSectors):
            F[j, :] = sum((Din[j * nCountries: (j + 1) * nCountries:1, :] / mTauPrev[j * nCountries: (j + 1) * nCountries:1, :]).T)

        mAlphas = num / (np.ones([nSectors, 1], dtype=float)).dot((VAn + sum(XO * (1 - F)).T.reshape(nCountries, 1) - Sn).T)
        for j in range(nSectors):
            for n in range(nCountries):
                if mAlphas[j, n] < 0:
                    mAlphas[j, n] = 0

        mAlphas = mAlphas / np.ones([nSectors, 1]).dot(sum(mAlphas).reshape(1, nCountries))
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
        mAllPrice = np.zeros([nYears * nSectors, nCountries], dtype=float)
        # ============================================================================================
        # Routine that repeat for nYears years
        # ============================================================================================
        for nActualYear in range(nYears):

            print("Running for Year = ", nActualYear)
            TInicio = time.perf_counter()
            print("Begin: ", time.strftime("%d/%b/%Y - %H:%M:%S", time.localtime()))
            LG = np.ones([nSectors, 1], dtype=float)
            for j in range(nSectors):
                LG[j, 0] = mGrowthLabor[nActualYear, j + 1]

            if nActualYear == 0:  # First Year
                mTauHat = mTauActual / mTauPrev
            else:
                mTauHat = mTauActual / mTauActual

            mWages, mPriceFactor, PQ, mWeightedTariffs, mTradeShare, ZW, Snp2, mCost, DP, PF, mWagesBrasil \
                = equilibrium_LC(mTauHat, mTauActual, mAlphas, mLinearThetas, mThetas, mShareVA, mIO, Din, nSectors,
                                 nCountries, nMaxIterations, nTolerance, VAn, Sn, vFactor, LG, VA_Br, nBeta,
                                 nPositionBR, nActualYear, w_aux, wbr_aux, mCsiBrasil, Csi_teste, mIota)

            w_aux = np.ones([nCountries, nYears], dtype=float)
            wbr_aux = np.ones([nSectors, nYears], dtype=float)
            for n in range(nCountries):
                w_aux[n, nActualYear] = mWages[n, 0]

            for j in range(nSectors):
                wbr_aux[j, nActualYear] = mWagesBrasil[j, 0]

            PQ_vec = PQ.T.reshape(nSectors * nCountries, 1, order='F').copy()  # expenditures Xji in long vector: PQ_vec=(X11 X12 X13...)'
            Dinp_om = mTradeShare / mTauActual
            xbilattau = (PQ_vec.dot(np.ones((1, nCountries)))) * Dinp_om
            xbilatp = xbilattau * mTauActual
            for j in range(nSectors):
                mGrossOutput[j, :] = sum(xbilattau[j * nCountries: (j + 1) * nCountries, :])

            VAjn = mGrossOutput * mShareVA
            VAn = sum(VAjn).T.reshape(nCountries, 1)
            # dif no VA_Br 2/5/2018 00:51
            VA_Br = VAjn[:, nPositionBR].reshape(nSectors, 1)
            Din = mTradeShare
            for j in range(nSectors):
                VABrasil[nActualYear, j] = VA_Br[j, 0]
                w_Brasil[nActualYear, j] = mWagesBrasil[j, 0]
                P_Brasil[nActualYear, j] = mPriceFactor[j, nPositionBR]

            # pf0_all = mPriceFactor. / (mAlphas);
            # P = prod(pf0_all. ^ (mAlphas));
            # PBr(nActualYear, 1) = P(1, nPositionBR);
            # pf0_all = mPriceFactor./(mAlphas);
            P = np.prod(mPriceFactor ** mAlphas, axis=0)
            PBr[nActualYear, 0] = P[nPositionBR]
            # xbilatp_old = xbilatp.copy()
            # for j in range(nSectors):
            #     for n in range(nCountries):
            #         xbilatp_old[n + j * nCountries, n] = 0
            sidx = np.arange(nSectors)
            cidx = np.arange(nCountries)
            xbilatp[cidx + sidx[:,None] * nCountries, cidx] = 0
            # assert np.array_equal(xbilatp, xbilatp_old)


            # xbilat_total_old = xbilat_total.copy()
            # for i in range(nCountries * nSectors):
            #     for n in range(nCountries):
            #         xbilat_total_old[nActualYear * nCountries * nSectors + i, n] = xbilatp[i, n]
            n = nCountries * nSectors
            xbilat_total[nActualYear*n:(nActualYear+1)*n] = xbilatp
            # assert np.array_equal(xbilat_total, xbilat_total_old)

            # mGrossOutputTotal_old = mGrossOutputTotal.copy()
            # mAllPrice_old = mAllPrice.copy()
            # for j in range(nSectors):
            #     for n in range(nCountries):
            #         mGrossOutputTotal_old[nActualYear * nSectors + j, n] = mGrossOutput[j, n]
            #         mAllPrice_old[nActualYear * nSectors + j, n] = mPriceFactor[j, n]
            mGrossOutputTotal[nActualYear*nSectors:(nActualYear+1)*nSectors] = mGrossOutput
            mAllPrice[nActualYear*nSectors:(nActualYear+1)*nSectors] = mPriceFactor
            # assert np.array_equal(mGrossOutputTotal, mGrossOutputTotal_old)
            # assert np.array_equal(mAllPrice, mAllPrice_old)

            print("End    : ", time.strftime("%d/%b/%Y - %H:%M:%S", time.localtime()))
            TFim = time.perf_counter()
            TDecorrido = (TFim - TInicio)
            print("Spent: %.2f segs" % TDecorrido)

#        Y_aux = mY
        Y_aux = np.ones([nYears, nSectorsLabor], dtype=float)
        for i in range(nYears - 1, 0, -1):
            Y_aux[i - 1, 0] = np.dot(mMigration[(i - 1) * nSectorsLabor, :], (Y_aux[i, :] ** nBeta).T.reshape(nSectorsLabor, 1))
            for j in range(nSectors):
                Y_aux[i - 1, j + 1] = np.dot(((w_Brasil[i - 1, j] / PBr[i - 1]) ** (1 / nValIntertemp)),
                                             np.dot(mMigration[(i - 1) * nSectorsLabor + j + 1, :], (Y_aux[i, :] ** nBeta).T))

        Y1_ant = mY
        Y1 = Y_aux
        Y_aux2 = sum(abs(Y1 - mY))
        vYmax = np.zeros([1, 1], dtype=float)
        vYmax[0, 0] = sum(Y_aux2.T)
        Ymax = vYmax[0, 0]
        print( "Ymax = ", Ymax)
        Y2 = mY - nAdjust * (mY - Y1)
        if nCenario == 0:
            lDataToSave = ['Y1', 'w_aux', 'wbr_aux', 'Y1_ant', 'YmaxV']
        else:
            lDataToSave = ['Y1_C', 'w_aux_C', 'wbr_aux_C', 'Y1_ant_C', 'YmaxV_C']

        lData = [Y1, w_aux, wbr_aux, Y1_ant, vYmax]
        Support.write_data_csv(lDataToSave, lData, sDirectoryOutput)
        mY = Y2
        nIteration +=1

    return VABrasil, w_Brasil, P_Brasil, mY, mGrowthLabor, PBr, xbilat_total, mGrossOutputTotal, mAllPrice, mMigration
