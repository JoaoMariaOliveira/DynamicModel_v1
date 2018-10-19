# ============================================================================================
# Main Module of Dynamic Computable General Equilibrium Model
# ported to Python by João Maria de Oliveira -
# Original Model - cAliendo e Parro (2017)
# ============================================================================================
import yaml
import numpy as np
import SupportFunctions as Support
from Controle import Equilibrium
from multiprocessing import Pool, cpu_count
import time

conf = yaml.load(open('config.yaml', 'r'))

sDirectoryInput  = conf['sDirectoryInput']
sDirectoryOutput = conf['sDirectoryOutput']
nFactor = conf['nFactor']
nTolerance = conf['nTolerance']
nMaxIteration = conf['nMaxIteration']
nAdjust = conf['nAdjust']
nSectors = conf['nSectors']
nTradebleSectors = conf['nTradebleSectors']
nCountries = conf['nCountries']
M = conf['M']
nYears = conf['nYears']
nSectorsLabor = nSectors+1
nPositionBR = conf['nPositionBR']
nBeta = conf['nBeta']
nValIntertemp = conf['nValIntertemp']

# ============================================================================================
# read follows data of Brazil in txt files:
# L         - labor Vector stock by GTAP sector + unemployment ( 0 pos)
# migracao  - migracao Matrix by GTAP sector + unemployment
# Csi       - Csi Matrix by GTAP sector - Percentual do Capital no VA letra grega CI
# ============================================================================================
mInitialLaborStock = Support.read_file_txt('L', sDirectoryInput)
mInitialMigration  = Support.read_file_txt('migracao', sDirectoryInput)
mCsiBrasil         = Support.read_file_txt('Csi', sDirectoryInput).reshape(nSectors, 1)


def run_scenario(scenario):
    # Assume normal scenario name is 'normal'
    isNormal = scenario['name'] == 'normal'

    # generation of initial values (from the beginning or from previously saved data in an iteration)
    if scenario['nStart'] == 0:
        Y1 = np.ones([nYears, nSectorsLabor], dtype=float)
        w_aux = np.ones([nCountries, nYears], dtype=float)
        wbr_aux = np.ones([nSectors, nYears], dtype=float)
        lData = [Y1, w_aux, wbr_aux]
        if isNormal:
            lDataToSave = ['Y1', 'w_aux', 'wbr_aux']
        else:
            lDataToSave = ['Y1_C', 'w_aux_C', 'wbr_aux_C']
        Support.write_data_csv(lDataToSave, lData, sDirectoryOutput)
    else:
        Y1 = Support.read_file_csv(scenario['Y1_input'], sDirectoryOutput)
        Y1_ant = Support.read_file_csv(scenario['Y1_ant_input'], sDirectoryOutput)
        Y2 = Y1_ant - 1 * (Y1_ant - Y1)
        Y1 = Y2

    mInitialY = Y1

    if isNormal:
        lSheet = ['VABrasil', 'w_Brasil', 'P_Brasil', 'Y', 'cresc_trab', 'PBr',
                  'xbilat_total', 'GO_total', 'p_total', 'migr']
    else:
        lSheet = ['VABrasil_C', 'w_Brasil_C', 'P_Brasil_C', 'Y_C', 'cresc_trab_C', 'PBr_C',
                  'xbilat_total_C', 'GO_total_C', 'p_total_C', 'migr_C']

    if scenario['nExecute'] == 0:
        sDirectoryInputScenario = scenario['sDirectoryInputScenario']
        sNameScenario = scenario['name']
        results = Equilibrium(nCountries, nSectors, nTradebleSectors, nSectorsLabor, nYears, nBeta, nValIntertemp,
                                nPositionBR, mInitialMigration, mInitialLaborStock, nFactor, nMaxIteration, nTolerance,
                                mInitialY, nAdjust, mCsiBrasil, isNormal, sDirectoryInputScenario, sDirectoryOutput, sNameScenario)
        sFileName = scenario['output']
        lData     = list(results)
        Support.write_data_excel(sDirectoryOutput, sFileName, lSheet, lData)
        return results
    else:
        return Support.read_data_excel(sDirectoryOutput, scenario['output'], lSheet)


if __name__ == '__main__':

    nBeginModel = time.perf_counter()
    sTimeBeginModel = time.localtime()
    # Assuming first scenario is normal
    normal_scenario = conf['scenarios'][0]
    counter_scenarios = conf['scenarios'][1:]
    nNumberScenarios = len(conf['scenarios'])
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Running Model for ", nYears, "Years to ", nSectors, " Sect. x ", nCountries, " Coun. and Tolerance = ",
          nTolerance)
    print(nNumberScenarios, "Scenarios : ", end=" ")
    for i in range(nNumberScenarios):
        if (i == 0):
           print("Normal", end=" ")
        else:
            print(",", counter_scenarios[i-1]['name'], end=" ")

    print(" ")
    print("Begin at ", time.strftime("%d/%b/%Y - %H:%M:%S",sTimeBeginModel ))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    if conf['parallel']:
        # Use either max available cores,
        # or enough to cover all scenarios
        cores = min(cpu_count(), nNumberScenarios)
        p = Pool(cores)

        normal_res = p.apply_async(run_scenario, (normal_scenario,))

        # Run other scenarios
        shock_res = [(scenario['name'], p.apply_async(run_scenario, (scenario,))) for scenario in counter_scenarios]

        VABrasil_pre, w_Brasil_pre, mPricesBrazilNorm, mYNorm, mGrowthLaborNorm, PBr_pre, xbilat_total_pre, \
            mGrossOutputTotalNorm, mAllPriceNorm, mMigrationNorm, sDirectoryInputScenarioNom, mTauNorm,\
            mAlphasNorm  = normal_res.get(timeout=None)
        shock_res = [(name, res.get(timeout=None)) for (name, res) in shock_res]
        p.close()
    else:
        VABrasil_pre, w_Brasil_pre, mPricesBrazilNorm, mYNorm, mGrowthLaborNorm, PBr_pre, xbilat_total_pre, \
            mGrossOutputTotalNorm, mAllPriceNorm, mMigrationNorm, sDirectoryInputScenarioNom, mTauNorm,\
            mAlphasNorm = run_scenario(normal_scenario)
        shock_res = [(scenario['name'], run_scenario(scenario)) for scenario in counter_scenarios]

    # Compare normal against each counterfactual

    nEndModel = time.perf_counter()
    nElapsedTime = (nEndModel - nBeginModel)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Model for ", nYears, "Years to ", nSectors," Sect. x ", nCountries, " Coun. and Tolerance = ",nTolerance)
    print("Begin at ",time.strftime("%d/%b/%Y - %H:%M:%S",sTimeBeginModel )
          ," End at ", time.strftime("%d/%b/%Y - %H:%M:%S", time.localtime()), "Spent: %.2f segs" % nElapsedTime )
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    for scenario_name, res in shock_res:
        VABrasil_pos, w_Brasil_pos, mPricesBrazilShock, mYShock, mGrowthLaborShock, PBr_pos, xbilat_total_pos, \
            mGrossOutputTotalShock, mAllPriceShock, mMigrationShock, sDirectoryInputScenario, mTau, mAlphas = res

        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("Data Treatment of Counterfactual Scenario ", scenario_name )
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        # ============================================================================================
        # Treatment of data for presentation of results
        # ============================================================================================
        vDataSheet = []
        vSheetName = []
        # ============================================================================================
        # variation of Prices
        # ============================================================================================
        p_pre = np.zeros([nYears, nSectors], dtype=float)
        p_pos = np.zeros([nYears, nSectors], dtype=float)
        p_pre[0, :] = mPricesBrazilNorm[0, :]
        p_pos[0, :] = mPricesBrazilShock[0, :]
        for t in range(1,nYears):
            for j in range(nSectors):
                p_pre[t, j] = mPricesBrazilNorm[t, j] * p_pre[t - 1, j]
                p_pos[t, j] = mPricesBrazilShock[t, j] * p_pos[t - 1, j]

        Crescimento_Preco = (p_pos / p_pre - 1) * 100
        vDataSheet.append(Crescimento_Preco)
        vSheetName.append('VariacaoPrecos')
        # ============================================================================================
        # Price Index Variation
        # ============================================================================================
        IP_pre = np.zeros([nYears], dtype=float)
        IP_pos = np.zeros([nYears], dtype=float)

        IP_pre[0] = PBr_pre[0]
        IP_pos[0] = PBr_pos[0]

        for t in range(1,nYears):
            IP_pre[t] = PBr_pre[t] * IP_pre[t - 1]
            IP_pos[t] = PBr_pos[t] * IP_pos[t - 1]
        Crescimento_Indice_Preco = ((IP_pos / IP_pre - 1) * 100).reshape(nYears, 1)
        vDataSheet.append(Crescimento_Indice_Preco)
        vSheetName.append('IndicePrecos')
        # ============================================================================================
        # Variation of wages
        # ============================================================================================
        S_pre = np.zeros([nYears, nSectors], dtype=float)
        S_pos = np.zeros([nYears, nSectors], dtype=float)
        S_pre[0, :] = w_Brasil_pre[0, :]
        S_pos[0, :] = w_Brasil_pos[0, :]
        for t in range(1,nYears):
            for j in range(nSectors):
                S_pre[t, j] = w_Brasil_pre[t, j] * S_pre[t - 1, j]
                S_pos[t, j] = w_Brasil_pos[t, j] * S_pos[t - 1, j]

        IP_pre_aux = np.tile(IP_pre.reshape(nYears,1), nSectors)
        S_pre = S_pre / IP_pre_aux
        IP_pos_aux = np.tile(IP_pos.reshape(nYears,1), nSectors)
        S_pos = S_pos / IP_pos_aux
        Crescimento_Salario = (S_pos / S_pre - 1) * 100
        vDataSheet.append(Crescimento_Salario)
        vSheetName.append('VariacaoSalarios')
        # ============================================================================================
        #
        # ============================================================================================
        VA_pre = VABrasil_pre / p_pre
        VA_pos = VABrasil_pos / p_pos
        VA_total_pre = (sum(VA_pre.T)).reshape(nYears,1)
        VA_total_pos = (sum(VA_pos.T)).reshape(nYears,1)
        VA_pre = np.hstack((VA_pre, VA_total_pre))
        VA_pos = np.hstack((VA_pos, VA_total_pos))
        Crescimento = (VA_pos / VA_pre - 1) * 100
        vDataSheet.append(Crescimento)
        vSheetName.append('VariacaoPIB')
        # ============================================================================================
        # Change in GDP (growth)
        # ============================================================================================
        VA_agricultura_pre = np.zeros([nYears], dtype=float)
        VA_agricultura_pos = np.zeros([nYears], dtype=float)
        VA_ind_extr_pre    = np.zeros([nYears], dtype=float)
        VA_ind_extr_pos    = np.zeros([nYears], dtype=float)
        VA_ind_tran_pre    = np.zeros([nYears], dtype=float)
        VA_ind_tran_pos    = np.zeros([nYears], dtype=float)
        VA_serv_pre        = np.zeros([nYears], dtype=float)
        VA_serv_pos        = np.zeros([nYears], dtype=float)
        for t in range(nYears):
            VA_agricultura_pre[t] = sum(VA_pre[t, 0:14])
            VA_agricultura_pos[t] = sum(VA_pos[t, 0:14])
            VA_ind_extr_pre[t]    = sum(VA_pre[t, 14:18])
            VA_ind_extr_pos[t]    = sum(VA_pos[t, 14:18])
            VA_ind_tran_pre[t]    = sum(VA_pre[t, 18:42])
            VA_ind_tran_pos[t]    = sum(VA_pos[t, 18:42])
            VA_serv_pre[t]        = sum(VA_pre[t, 42:57])
            VA_serv_pos[t]        = sum(VA_pos[t, 42:57])

        VA_setores_pre = np.concatenate((VA_agricultura_pre.reshape(nYears,1), VA_ind_extr_pre.reshape(nYears,1), VA_ind_tran_pre.reshape(nYears,1), VA_serv_pre.reshape(nYears,1)),axis=1)
        VA_setores_pos = np.concatenate((VA_agricultura_pos.reshape(nYears,1), VA_ind_extr_pos.reshape(nYears,1), VA_ind_tran_pos.reshape(nYears,1), VA_serv_pos.reshape(nYears,1)),axis=1)
        Crescimento_setores = (VA_setores_pos / VA_setores_pre - 1) * 100
        vDataSheet.append(Crescimento_setores)
        vSheetName.append('VariacaoPIBSetorial')
        # ============================================================================================
        # Exchange Variation
        # ============================================================================================
        GO_pre_time = np.zeros([nSectors, nCountries], dtype=float)
        GO_pos_time = np.zeros([nSectors, nCountries], dtype=float)
        p_pre_time = np.zeros([nSectors, nCountries], dtype=float)
        p_pos_time = np.zeros([nSectors, nCountries], dtype=float)
        tau_pre_time = np.zeros([nSectors * nCountries, nCountries], dtype=float)
        tau_pos_time = np.zeros([nSectors * nCountries, nCountries], dtype=float)
#        tau = np.vstack((1 + Tarifas / 100, np.ones([15 * nCountries, nCountries], dtype=float)))
#        tau_pre = np.tile(tau, (nYears, 1))
#        tau = np.vstack((1 + TarifasZero / 100, np.ones([15 * nCountries, nCountries], dtype=float)))
#        tau_pos = np.tile(tau, (nYears, 1))
        GO_totalaux_pre = np.zeros([nSectors, nCountries], dtype=float)
        GO_totalaux_pos = np.zeros([nSectors, nCountries], dtype=float)
        tau_pre_aux = np.zeros([nCountries * nTradebleSectors, nCountries], dtype=float)
        tau_pos_aux = np.zeros([nCountries * nTradebleSectors, nCountries], dtype=float)
        total_comex_pre = np.zeros([nCountries, nCountries], dtype=float)
        total_comex_pos = np.zeros([nCountries, nCountries], dtype=float)
        Cambio = np.zeros([nYears, nCountries], dtype=float)
        for nActualYear in range(nYears):

            for j in range(nSectors):
                for n in range(nCountries):
                    GO_pre_time[j, n] = mGrossOutputTotalNorm[nActualYear * nSectors + j, n]
                    GO_pos_time[j, n] = mGrossOutputTotalShock[nActualYear * nSectors + j, n]
                    p_pre_time[j, n] = mAllPriceNorm[nActualYear * nSectors + j, n]
                    p_pos_time[j, n] = mAllPriceShock[nActualYear * nSectors + j, n]

            export_pre = np.zeros([nSectors * nCountries, nCountries], dtype=float)
            export_pos = np.zeros([nSectors * nCountries, nCountries], dtype=float)
            for i in range(nSectors * nCountries):
                export_pre[i, :] = xbilat_total_pre[nActualYear * nSectors * nCountries + i, :]
                export_pos[i, :] = xbilat_total_pos[nActualYear * nSectors * nCountries + i, :]
#                    tau_pre_time[i, n] = tau_pre[nActualYear * nSectors * nCountries + i, n]
#                    tau_pos_time[i, n] = tau_pos[nActualYear * nSectors * nCountries + i, n]
                tau_pre_time[i, :] = mTauNorm[nActualYear * nSectors * nCountries + i, :]
                tau_pos_time[i, :] = mTau[nActualYear * nSectors * nCountries + i, :]


            GO_aux_pre = sum(GO_pre_time)
            GO_aux_pos = sum(GO_pos_time)

            for j in range(nSectors):
                for n in range(nCountries):
                    GO_totalaux_pre[j, n] = GO_aux_pre[n]
                    GO_totalaux_pos[j, n] = GO_aux_pos[n]

            GO_pesos_pre = GO_pre_time / GO_totalaux_pre
            GO_pesos_pos = GO_pos_time / GO_totalaux_pos
            ind_p_pre = np.prod(p_pre_time ** GO_pesos_pre, axis=0)
            ind_p_pos = np.prod(p_pos_time ** GO_pesos_pos, axis=0)
            export_pre_aux = np.zeros([nCountries * nTradebleSectors, nCountries], dtype=float)
            export_pos_aux = np.zeros([nCountries * nTradebleSectors, nCountries], dtype=float)
            for i in range(nCountries * nTradebleSectors):
                export_pre_aux[i, :] = export_pre[i, :]
                export_pos_aux[i, :] = export_pos[i, :]
                tau_pre_aux[i, :] = tau_pre_time[i, :]
                tau_pos_aux[i, :] = tau_pos_time[i, :]

            del export_pre, export_pos
            export_pre = export_pre_aux / tau_pre_aux
            export_pos = export_pos_aux / tau_pos_aux
            del export_pre_aux, export_pos_aux
            export_pre_aux = np.zeros([M , nCountries], dtype=float)
            export_pos_aux = np.zeros([M , nCountries], dtype=float)
            for n in range(nCountries):
                for m in range(M):
                    export_pre_aux[m, n] = sum(export_pre[m: M * nTradebleSectors:M, n]).T
                    export_pos_aux[m, n] = sum(export_pos[m: M * nTradebleSectors:M, n]).T

            sum_export_pre = sum(export_pre_aux.T)
            sum_export_pos = sum(export_pos_aux.T)

            sum_import_pre = sum(export_pre_aux)
            sum_import_pos = sum(export_pos_aux)

            comex_pre = sum_export_pre + sum_import_pre
            comex_pos = sum_export_pos + sum_import_pos
            for i in range(nCountries):
                for n in range(nCountries):
                    total_comex_pre[i, n] = (export_pre_aux[i, n] + export_pre_aux[n, i])
                    total_comex_pos[i, n] = (export_pos_aux[i, n] + export_pos_aux[n, i])

            comex_pre = np.tile(comex_pre, (nCountries, 1))
            comex_pos = np.tile(comex_pos, (nCountries, 1))
            pesos_comex_pre = total_comex_pre / comex_pre
            pesos_comex_pos = total_comex_pos / comex_pos
            ind_p_pre_aux = np.tile(ind_p_pre.T.reshape(nCountries,1), (1, nCountries))
            ind_p_pos_aux = np.tile(ind_p_pos.T.reshape(nCountries,1), (1, nCountries))
            p_exterior_pre = np.prod(ind_p_pre_aux ** pesos_comex_pre, axis=0)
            p_exterior_pos = np.prod(ind_p_pos_aux ** pesos_comex_pos, axis=0)
            cambio_pre = p_exterior_pre / ind_p_pre
            cambio_pos = p_exterior_pos / ind_p_pos
            for n in range(nCountries):
                Cambio[nActualYear, n] = cambio_pos[n] / cambio_pre[n]

        Crescimento_cambio = np.zeros([nYears, nCountries], dtype=float)
        Crescimento_cambio[0, :] = Cambio[0, :]
        for nActualYear in range(1, nYears):
            Crescimento_cambio[nActualYear, :] = Cambio[nActualYear, :] * Crescimento_cambio[nActualYear - 1, :]

        vDataSheet.append(Crescimento_cambio)
        vSheetName.append('VariacaoCambial')

        # ============================================================================================
        # Variation of sector-country exports and Brasil exports by sector
        # ============================================================================================
        tau_pre_time   = np.zeros([nSectors * nCountries, nCountries], dtype=float)
        tau_pos_time   = np.zeros([nSectors * nCountries, nCountries], dtype=float)
        tau_pre_aux    = np.zeros([nCountries * nTradebleSectors, nCountries], dtype=float)
        tau_pos_aux    = np.zeros([nCountries * nTradebleSectors, nCountries], dtype=float)
        cresc_export_brasil = np.zeros([nYears, nTradebleSectors], dtype=float)
        export_brasil_pre   = np.zeros([nYears, nTradebleSectors], dtype=float)
        export_brasil_pos   = np.zeros([nYears, nTradebleSectors], dtype=float)
        cresc_export        = np.zeros([nYears*nSectors, nCountries], dtype=float)
        cresc_export_total  = np.zeros([nYears, nCountries], dtype=float)
        export_brasil_pre_aux = np.zeros([nYears*nSectors*nTradebleSectors, nCountries], dtype=float)
        export_brasil_pos_aux = np.zeros([nYears*nSectors*nTradebleSectors, nCountries], dtype=float)
        for nActualYear in range(nYears):

            export_pre = np.zeros([nSectors * nCountries, nCountries], dtype=float)
            export_pos = np.zeros([nSectors * nCountries, nCountries], dtype=float)
            for i in range(nSectors * nCountries):
                for n in range(nCountries):
                    export_pre[i, n] = xbilat_total_pre[nActualYear * nSectors * nCountries + i, n]
                    export_pos[i, n] = xbilat_total_pos[nActualYear * nSectors * nCountries + i, n]
                    tau_pre_time[i, n] = mTauNorm[nActualYear * nSectors * nCountries + i, n]
                    tau_pos_time[i, n] = mTau[nActualYear * nSectors * nCountries + i, n]

            export_pre_aux = np.zeros([nCountries * nTradebleSectors, nCountries], dtype=float)
            export_pos_aux = np.zeros([nCountries * nTradebleSectors, nCountries], dtype=float)
            for i in range(nCountries * nTradebleSectors):
                export_pre_aux[i, :] = export_pre[i, :]
                export_pos_aux[i, :] = export_pos[i, :]
                tau_pre_aux[i, :] = tau_pre_time[i, :]
                tau_pos_aux[i, :] = tau_pos_time[i, :]

            del export_pre, export_pos
            export_pre = export_pre_aux / tau_pre_aux
            export_pos = export_pos_aux / tau_pos_aux

            del export_pre_aux, export_pos_aux
            export_pre_aux = np.zeros([nTradebleSectors, nCountries], dtype=float)
            export_pos_aux = np.zeros([nTradebleSectors, nCountries], dtype=float)

            for j in range(nTradebleSectors):
                for n in range(nCountries):
                    # Exports  1+nCountries*(j-1):nCountries*j,n)
                    export_pre_aux[j, n] = sum(export_pre[nCountries*j:(nCountries*(j+1)), n]).T
                    # Exports
                    export_pos_aux[j, n] = sum(export_pos[nCountries*j:(nCountries*(j+1)), n]).T

                    cresc_export[nActualYear * nSectors + j, n] = export_pos_aux[j, n] / export_pre_aux[j, n]

                    export_brasil_pre_aux[nActualYear * nSectors + j, n] = export_pre_aux[j, n]
                    export_brasil_pos_aux[nActualYear * nSectors + j, n] = export_pos_aux[j, n]

            export_pre_soma = sum(export_pre_aux)
            export_pos_soma = sum(export_pos_aux)
            for n in range(nCountries):
                cresc_export_total[nActualYear, n] = export_pos_soma[n] / export_pre_soma[n]

        for nActualYear in range(nYears):
            for i in range (nTradebleSectors):
                cresc_export_brasil[nActualYear,i] = cresc_export[nActualYear*nSectors+i,nPositionBR]
                export_brasil_pre[nActualYear,i] = export_brasil_pre_aux[nActualYear*nSectors+i,nPositionBR]
                export_brasil_pos[nActualYear,i] = export_brasil_pos_aux[nActualYear*nSectors+i,nPositionBR]

        vDataSheet.append(cresc_export_total)
        vSheetName.append('CrescExportTotal')
        vDataSheet.append(cresc_export_brasil)
        vSheetName.append('CrescExportBrasil')
        # ============================================================================================
        # Variation in sectoral exports
        # ============================================================================================
        export_agricultura_pre = np.zeros([nYears], dtype=float)
        export_agricultura_pos = np.zeros([nYears], dtype=float)
        export_ind_extr_pre    = np.zeros([nYears], dtype=float)
        export_ind_extr_pos    = np.zeros([nYears], dtype=float)
        export_ind_tran_pre    = np.zeros([nYears], dtype=float)
        export_ind_tran_pos    = np.zeros([nYears], dtype=float)
        for t in range(nYears):
            export_agricultura_pre[t] = np.sum(export_brasil_pre[t, 0:14], axis=0)
            export_agricultura_pos[t] = np.sum(export_brasil_pos[t, 0:14], axis=0)
            export_ind_extr_pre[t]    = np.sum(export_brasil_pre[t, 14:18], axis=0)
            export_ind_extr_pos[t]    = np.sum(export_brasil_pos[t, 14:18], axis=0)
            export_ind_tran_pre[t]    = np.sum(export_brasil_pre[t, 18:42], axis=0)
            export_ind_tran_pos[t]    = np.sum(export_brasil_pos[t, 18:42], axis=0)

        export_setores_pre = np.concatenate((export_agricultura_pre.reshape(nYears,1), export_ind_extr_pre.reshape(nYears,1), export_ind_tran_pre.reshape(nYears,1)), axis=1)
        export_setores_pos = np.concatenate((export_agricultura_pos.reshape(nYears,1), export_ind_extr_pos.reshape(nYears,1), export_ind_tran_pos.reshape(nYears,1)), axis=1)
        Crescimento_export_setores = (export_setores_pos / export_setores_pre-1)*100
        vDataSheet.append(Crescimento_export_setores)
        vSheetName.append('ExportacaoSetorial')
        # ============================================================================================
        # Variation of exports country x country
        # ============================================================================================
        export_pre = np.zeros([nSectors * nCountries, nCountries], dtype=float)
        export_pos = np.zeros([nSectors * nCountries, nCountries], dtype=float)
        GO_pre_time = np.zeros([nSectors, nCountries], dtype=float)
        GO_pos_time = np.zeros([nSectors, nCountries], dtype=float)
        p_pre_time = np.zeros([nSectors, nCountries], dtype=float)
        p_pos_time = np.zeros([nSectors, nCountries], dtype=float)
        tau_pre_time = np.zeros([nSectors * nCountries, nCountries], dtype=float)
        tau_pos_time = np.zeros([nSectors * nCountries, nCountries], dtype=float)
        for i in range(nSectors * nCountries):
            for n in range(nCountries):
                export_pre[i, n] = xbilat_total_pre[(nYears-2)* nSectors * nCountries + i, n]
                export_pos[i, n] = xbilat_total_pos[(nYears-2) * nSectors * nCountries + i, n]
                tau_pre_time[i, n] = mTauNorm[(nYears-2) * nSectors * nCountries + i, n]
                tau_pos_time[i, n] = mTau[(nYears-2) * nSectors * nCountries + i, n]

        export_pre_aux = np.zeros([nCountries * nTradebleSectors, nCountries], dtype=float)
        export_pos_aux = np.zeros([nCountries * nTradebleSectors, nCountries], dtype=float)
        for i in range(nCountries * nTradebleSectors):
            export_pre_aux[i, :] = export_pre[i, :]
            export_pos_aux[i, :] = export_pos[i, :]
            tau_pre_aux[i, :] = tau_pre_time[i, :]
            tau_pos_aux[i, :] = tau_pos_time[i, :]

        del export_pre, export_pos
        export_pre = export_pre_aux / tau_pre_aux
        export_pos = export_pos_aux / tau_pos_aux
        del export_pre_aux, export_pos_aux
        export_pre_aux = np.zeros([M, nCountries], dtype=float)
        export_pos_aux = np.zeros([M, nCountries], dtype=float)
        for n in range(nCountries):
            for m in range(M):
                # Exports
                export_pre_aux[m, n] = sum(export_pre[m: M * nTradebleSectors:M, n]).T
                # Exports
                export_pos_aux[m, n] = sum(export_pos[m: M * nTradebleSectors:M, n]).T

        Crescimento_export_por_pais = (export_pos_aux / export_pre_aux-1)*100
        for n in range (nCountries):
            Crescimento_export_por_pais[n,n] = 0

        vDataSheet.append(Crescimento_export_por_pais)
        vSheetName.append('ExportacaoPorPais')
        # ============================================================================================
        # Variation of exports Brazil x country x sector
        # ============================================================================================
        export_por_produto_pais_pos = (export_pos / export_pre - 1)*100
        export_por_produto_pais_pos[np.isnan(export_por_produto_pais_pos)]=0
        export_Brasil_pos = export_por_produto_pais_pos[:, nPositionBR]
        export_Brasil_final = np.zeros([nTradebleSectors, nCountries], dtype=float)
        for n in range(nCountries):
            export_Brasil_final[:,n] = export_Brasil_pos[n:nCountries*nTradebleSectors:nCountries]

        vDataSheet.append(export_Brasil_final)
        vSheetName.append('ExportacaoBrasil')
        # ============================================================================================
        # Variation of imports Brazil x country x sector
        # ============================================================================================
        import_Brasil_final = np.zeros([nTradebleSectors, nCountries], dtype=float)
        for n in range(nCountries):
            import_Brasil_final[:,n] = export_por_produto_pais_pos[nPositionBR:nCountries*nTradebleSectors:nCountries,n]

        vDataSheet.append(import_Brasil_final)
        vSheetName.append('ImportacaoBrasil')
        # ============================================================================================
        # variation in Labor Market
        # ============================================================================================
        num_trab_pre = np.zeros([nYears, nSectorsLabor], dtype=float)
        num_trab_pos = np.zeros([nYears, nSectorsLabor], dtype=float)
        num_trab_pre[0,:] = mInitialLaborStock * mGrowthLaborNorm[0, :]
        num_trab_pos[0,:] = mInitialLaborStock * mGrowthLaborShock[0, :]
        for t in range(1,nYears):
            num_trab_pre[t, :] = num_trab_pre[t-1,:] * mGrowthLaborNorm[t, :]
            num_trab_pos[t, :] = num_trab_pos[t-1,:] * mGrowthLaborShock[t, :]

        Cresc_PO = (num_trab_pos / num_trab_pre-1)*100
        vDataSheet.append(Cresc_PO)
        vSheetName.append('VariacaoPO')
        num_trab_aux_pre = num_trab_pre[:, 1:nSectors+1]
        num_trab_aux_pos = num_trab_pos[:, 1:nSectors+1]
        PO_agricultura_pre = np.zeros([nYears], dtype=float)
        PO_agricultura_pos = np.zeros([nYears], dtype=float)
        PO_ind_extr_pre    = np.zeros([nYears], dtype=float)
        PO_ind_extr_pos    = np.zeros([nYears], dtype=float)
        PO_ind_tran_pre    = np.zeros([nYears], dtype=float)
        PO_ind_tran_pos    = np.zeros([nYears], dtype=float)
        PO_serv_pre    = np.zeros([nYears], dtype=float)
        PO_serv_pos    = np.zeros([nYears], dtype=float)
        for t in range(nYears):
            PO_agricultura_pre[t] = np.sum(num_trab_aux_pre[t, 0:14], axis=0)
            PO_agricultura_pos[t] = np.sum(num_trab_aux_pos[t, 0:14], axis=0)
            PO_ind_extr_pre[t]    = np.sum(num_trab_aux_pre[t, 14:18], axis=0)
            PO_ind_extr_pos[t]    = np.sum(num_trab_aux_pos[t, 14:18], axis=0)
            PO_ind_tran_pre[t]    = np.sum(num_trab_aux_pre[t, 18:42], axis=0)
            PO_ind_tran_pos[t]    = np.sum(num_trab_aux_pos[t, 18:42], axis=0)
            PO_serv_pre[t]        = np.sum(num_trab_aux_pre[t, 42:57], axis=0)
            PO_serv_pos[t]        = np.sum(num_trab_aux_pos[t, 42:57], axis=0)

        PO_setores_pre = np.concatenate((PO_agricultura_pre.reshape(nYears,1), PO_ind_extr_pre.reshape(nYears,1), PO_ind_tran_pre.reshape(nYears,1), PO_serv_pre.reshape(nYears,1)), axis=1)
        PO_setores_pos = np.concatenate((PO_agricultura_pos.reshape(nYears,1), PO_ind_extr_pos.reshape(nYears,1), PO_ind_tran_pos.reshape(nYears,1), PO_serv_pos.reshape(nYears,1)), axis=1)
        Crescimento_PO_setores = (PO_setores_pos /PO_setores_pre-1)*100
        vDataSheet.append(Crescimento_PO_setores)
        vSheetName.append('POSetorial')
        # ============================================================================================
        # variation of productivity total and by sector
        # ============================================================================================
        num_trab_aux2_pre = np.sum(num_trab_aux_pre,axis=1)
        num_trab_aux2_pre = np.concatenate((num_trab_aux_pre, num_trab_aux2_pre.reshape(nYears,1)), axis=1)
        num_trab_aux2_pos = np.sum(num_trab_aux_pos,axis=1)
        num_trab_aux2_pos = np.concatenate((num_trab_aux_pos, num_trab_aux2_pos.reshape(nYears,1)), axis=1)
        produtividade_pre = VA_pre / num_trab_aux2_pre
        produtividade_pos = VA_pos / num_trab_aux2_pos
        Crescimento_produtividade = (produtividade_pos/produtividade_pre-1)*100
        vDataSheet.append(Crescimento_produtividade)
        vSheetName.append('Produtividade')
        prod_agricultura_pre = VA_agricultura_pre / PO_agricultura_pre
        prod_agricultura_pos = VA_agricultura_pos / PO_agricultura_pos
        prod_ind_extr_pre = VA_ind_extr_pre / PO_ind_extr_pre
        prod_ind_extr_pos = VA_ind_extr_pos / PO_ind_extr_pos
        prod_ind_tran_pre = VA_ind_tran_pre / PO_ind_tran_pre
        prod_ind_tran_pos = VA_ind_tran_pos / PO_ind_tran_pos
        prod_serv_pre = VA_serv_pre / PO_serv_pre
        prod_serv_pos = VA_serv_pos / PO_serv_pos
        prod_setores_pre = np.concatenate((prod_agricultura_pre.reshape(nYears,1), prod_ind_extr_pre.reshape(nYears,1), prod_ind_tran_pre.reshape(nYears,1), prod_serv_pre.reshape(nYears,1)), axis=1)
        prod_setores_pos = np.concatenate((prod_agricultura_pos.reshape(nYears,1), prod_ind_extr_pos.reshape(nYears,1), prod_ind_tran_pos.reshape(nYears,1), prod_serv_pos.reshape(nYears,1)), axis=1)
        Cresc_produtividade_setores = (prod_setores_pos / prod_setores_pre-1)*100
        vDataSheet.append(Cresc_produtividade_setores)
        vSheetName.append('ProdutividadeSetorial')
        # ============================================================================================
        # Variation of adjustment of costs
        # ============================================================================================


        # ============================================================================================
        # Variation of sectoral wages
        # ============================================================================================
        Sal_agricultura_pre = np.zeros([nYears], dtype=float)
        Sal_agricultura_pos = np.zeros([nYears], dtype=float)
        Sal_ind_extr_pre    = np.zeros([nYears], dtype=float)
        Sal_ind_extr_pos    = np.zeros([nYears], dtype=float)
        Sal_ind_tran_pre    = np.zeros([nYears], dtype=float)
        Sal_ind_tran_pos    = np.zeros([nYears], dtype=float)
        Sal_serv_pre        = np.zeros([nYears], dtype=float)
        Sal_serv_pos        = np.zeros([nYears], dtype=float)
        Sal_economia_pre    = np.zeros([nYears], dtype=float)
        Sal_economia_pos    = np.zeros([nYears], dtype=float)
        for t in range(nYears):
            Sal_agricultura_pre[t] = np.sum(S_pre[t, 0:14] * num_trab_aux_pre[t, 0:14], axis=0) / PO_agricultura_pre[t]
            Sal_agricultura_pos[t] = np.sum(S_pos[t, 0:14] * num_trab_aux_pos[t,0:14], axis=0) / PO_agricultura_pos[t]
            Sal_ind_extr_pre[t]    = np.sum(S_pre[t, 14:18] * num_trab_aux_pre[t, 14:18], axis=0) / PO_ind_extr_pre[t]
            Sal_ind_extr_pos[t]    = np.sum(S_pos[t, 14:18] * num_trab_aux_pos[t, 14:18], axis=0) / PO_ind_extr_pos[t]
            Sal_ind_tran_pre[t]    = np.sum(S_pre[t, 18:42] * num_trab_aux_pre[t, 18:42], axis=0) / PO_ind_tran_pre[t]
            Sal_ind_tran_pos[t]    = np.sum(S_pos[t, 18:42] * num_trab_aux_pos[t, 18:42], axis=0) / PO_ind_tran_pos[t]
            Sal_serv_pre[t]        = np.sum(S_pre[t, 42:57] * num_trab_aux_pre[t, 42:57], axis=0) / PO_serv_pre[t]
            Sal_serv_pos[t]        = np.sum(S_pos[t, 42:57] * num_trab_aux_pos[t, 42:57], axis=0) / PO_serv_pos[t]
            Sal_economia_pre[t]    = np.sum(S_pre[t, 0:57] * num_trab_aux_pre[t, 0:57], axis=0) / np.sum(num_trab_aux_pre[t,0:57], axis=0)
            Sal_economia_pos[t]    = np.sum(S_pos[t, 0:57] * num_trab_aux_pos[t, 0:57], axis=0) / np.sum(num_trab_aux_pos[t,0:57], axis=0)

        Sal_setores_pre = np.concatenate((Sal_agricultura_pre.reshape(nYears,1), Sal_ind_extr_pre.reshape(nYears,1), Sal_ind_tran_pre.reshape(nYears,1), Sal_serv_pre.reshape(nYears,1), Sal_economia_pre.reshape(nYears,1)), axis=1)
        Sal_setores_pos = np.concatenate((Sal_agricultura_pos.reshape(nYears,1), Sal_ind_extr_pos.reshape(nYears,1), Sal_ind_tran_pos.reshape(nYears,1), Sal_serv_pos.reshape(nYears,1), Sal_economia_pos.reshape(nYears,1)), axis=1)
        Crescimento_Salario_setores = (Sal_setores_pos / Sal_setores_pre-1)*100
        vDataSheet.append(Crescimento_Salario_setores)
        vSheetName.append('SalariosSetoriais')
        # ============================================================================================
        # Variation of sectoral wages
        # ============================================================================================
        P_agricultura_pre = np.zeros([nYears], dtype=float)
        P_agricultura_pos = np.zeros([nYears], dtype=float)
        P_ind_extr_pre    = np.zeros([nYears], dtype=float)
        P_ind_extr_pos    = np.zeros([nYears], dtype=float)
        P_ind_tran_pre    = np.zeros([nYears], dtype=float)
        P_ind_tran_pos    = np.zeros([nYears], dtype=float)
        P_serv_pre        = np.zeros([nYears], dtype=float)
        P_serv_pos        = np.zeros([nYears], dtype=float)
        P_economia_pre    = np.zeros([nYears], dtype=float)
        P_economia_pos    = np.zeros([nYears], dtype=float)
        P_pre    = np.zeros([nYears,nSectors], dtype=float)
        P_pos    = np.zeros([nYears,nSectors], dtype=float)
        p_pre[0,:] = mPricesBrazilNorm[0, :]
        p_pos[0,:] = mPricesBrazilShock[0, :]
        for t in range(1,nYears):
            for j in  range(nSectors):
                p_pre[t, j] = mPricesBrazilNorm[t, j] * p_pre[t - 1, j]
                p_pos[t, j] = mPricesBrazilShock[t, j] * p_pos[t - 1, j]

        alphas_aux = mAlphas.T.reshape(nCountries, nSectors)

        for t in range(nYears):
            P_agricultura_pre[t] = np.prod(p_pre[t, 0:14] ** (alphas_aux[nPositionBR, 0:14]), axis=0)
            P_agricultura_pos[t] = np.prod(p_pos[t, 0:14]  ** (alphas_aux[nPositionBR, 0:14]), axis=0)
            P_ind_extr_pre[t]    = np.prod(p_pre[t, 14:18] ** (alphas_aux[nPositionBR, 14:18]), axis=0)
            P_ind_extr_pos[t]    = np.prod(p_pos[t, 14:18] ** (alphas_aux[nPositionBR, 14:18]), axis=0)

            P_ind_tran_pre[t]    = np.prod(p_pre[t, 18:42] ** (alphas_aux[nPositionBR, 18:42]), axis=0)
            P_ind_tran_pos[t]    = np.prod(p_pos[t, 18:42] ** (alphas_aux[nPositionBR, 18:42]), axis=0)
            P_serv_pre[t]        = np.prod(p_pre[t, 42:57] ** (alphas_aux[nPositionBR, 42:57]), axis=0)
            P_serv_pos[t]        = np.prod(p_pos[t, 42:57] ** (alphas_aux[nPositionBR, 42:57]), axis=0)

        Precos_setores_pre = np.concatenate((P_agricultura_pre.reshape(nYears,1), P_ind_extr_pre.reshape(nYears,1), P_ind_tran_pre.reshape(nYears,1), P_serv_pre.reshape(nYears,1)), axis=1)
        Precos_setores_pos = np.concatenate((P_agricultura_pos.reshape(nYears,1), P_ind_extr_pos.reshape(nYears,1), P_ind_tran_pos.reshape(nYears,1), P_serv_pos.reshape(nYears,1)), axis=1)

        Crescimento_Precos_setores = (Precos_setores_pos / Precos_setores_pre-1)*100
        vDataSheet.append(Crescimento_Precos_setores)
        vSheetName.append('PrecosSetoriais')
        # ============================================================================================
        # Adjustment Rate Variation of Labor stock
        # ============================================================================================
        num_trab_pre    = np.zeros([nYears, nSectorsLabor], dtype=float)
        num_trab_pos    = np.zeros([nYears, nSectorsLabor], dtype=float)
        cum_ajuste_aux  = np.zeros([nYears, nSectorsLabor], dtype=float)
        ajuste_PO       = np.zeros([nYears], dtype=float)
        num_trab_pre[0,:] = mInitialLaborStock * mGrowthLaborNorm[0, :]
        num_trab_pos[0,:] = mInitialLaborStock * mGrowthLaborShock[0, :]
        for t in range(1,nYears):
            num_trab_pre[t,:] = num_trab_pre[t-1,:] * mGrowthLaborNorm[t, :]
            num_trab_pos[t,:] = num_trab_pos[t-1,:] * mGrowthLaborShock[t, :]

        for t in range(nYears):
            cum_ajuste_aux[t,:] = .5 * abs(num_trab_pos[t,:]-num_trab_pre[t,:])

        cum_ajuste = sum(cum_ajuste_aux,1)
        for t in range(nYears):
            ajuste_PO[t] = 100*(cum_ajuste[t]/cum_ajuste[nYears-1])
        vDataSheet.append(ajuste_PO)
        vSheetName.append('AjustePO')
        # ============================================================================================
        # Variation of sectoral exports only to the counterfactual scenario
        # ============================================================================================


        # ============================================================================================
        # Variation of sectoral imports only to the counterfactual scenario
        # ============================================================================================


        # ============================================================================================
        # Recording ResultModel
        # ============================================================================================
        Support.write_data_excel(sDirectoryOutput, "ResultsOfModel_{}.xlsx".format(scenario_name), vSheetName, vDataSheet)

    print("End")