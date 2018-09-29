# ============================================================================================
# Modulo de controle do modelo dim^,mico crido por Messa
# portado para Python por João Maria de Oliveira - baseado no trabalho de Bernardo
# Modelo original de cAliendo e Parro (2017)
# ============================================================================================
#    PENDENCIAS
import numpy as np
import SupportFunctions as FuncoesApoio
from Controle import Equilibrium
import sys
# ============================================================================================
# parâmetros gerais
# ============================================================================================
# sDirectoryInput - Pasta de entrada de dados
# sDirectoryOutput - Pasta de Saída de dados
# sFileData          - Arquivo de dados
# sFileRate          - Arquivo
# sSheet             - Planilha de dados
sDirectoryInput  = './InputCenario1FormulaSuica/'
sDirectoryOutput = './OutputCenario1FormulaSuica/'

# ============================================================================================
# parâmetros de Excucação
# ============================================================================================
# nExecuta = 0 roda modelo e trata dados
# nExecuta = 1 trata dados (modelo já está rodado)
# posição 0 de nExecuta é para o modelo sem choque
# posição 1 de nExecuta é para o modelo com choque
nExecuta = [0, 0]
# nStart = 0 roda modelo  com Y inicial matriz de 1
# nStart = 1 roda modelo lendo Y inicial a partir de dados salvos anteriormente
# posição 0 de nStart é para o modelo sem choque
# posição 1 de nStart é para o modelo com choque
nStart = [0, 0]
# parâmetros iniciais
vfactor = -0.1
# valor de tolerância para convergência
#nTolerance = 1E-08
nTolerance = 1E-08
# Número máximo de iterações
nMaxIter = 400000000
nAdjust = 1
#
# Inputs
# Sectors J
nSectors = 57
# Tradebles  Sectors TS
nTradebleSectors = 42
# Countries N
nCountries = 27
M = 27
# Years T
nYears = 20
# Sectors + unemployment
D = nSectors+1
# posição do Brasil
Pos = 0

# ============================================================================================
# controle de chamada do CGE
# nChoque = 0 - roda sem choque
# nChoque = 1 - roda com choque
for nChoque in range(2):
    # geração de valores iniciais
    if nStart [nChoque] == 0:
        Y1 = np.ones([nYears, D], dtype=float)
        w_aux = np.ones([nCountries, nYears], dtype=float)
        p_aux = np.ones([nYears * nSectors, nCountries], dtype=float)
        wbr_aux = np.ones([nSectors, nYears], dtype=float)

        if nChoque == 0:
            lDataToSave = ['Y1', 'w_aux', 'wbr_aux']
        else:
            lDataToSave = ['Y1_C', 'w_aux_C', 'wbr_aux_C']

        lData = [Y1, w_aux, wbr_aux]
        FuncoesApoio.write_data_csv(lDataToSave, lData, sDirectoryOutput )
    else:
        if nChoque == 0:
            Y1 = FuncoesApoio.read_file_csv('Y1',sDirectoryOutput)
            Y1_ant = FuncoesApoio.read_file_csv('Y1_ant', sDirectoryOutput)
        else:
            Y1 = FuncoesApoio.read_file_csv('Y1_C', sDirectoryOutput)
            Y1_ant = FuncoesApoio.read_file_csv('Y1_ant_C', sDirectoryOutput)

        Y2 = Y1_ant - 1 * (Y1_ant - Y1)
        Y1 = Y2

    Yinic = Y1

    # Valores de calibragem
    beta = .9
    # valor intertemporal
    v = 8.47
# ============================================================================================
# read follows data of Brazil in txt files:
# L         - labor Vector stock by GTAP sector + unemployment ( 0 pos)
# migracao  - migracao Matrix by GTAP sector + unemployment
# Csi       - Csi Matrix by GTAP sector - Percentual do Capital no VA letra grega CI
# ============================================================================================
    L           = FuncoesApoio.read_file_txt('L',sDirectoryInput)
    migracao    = FuncoesApoio.read_file_txt('migracao',sDirectoryInput)
    Csi         = FuncoesApoio.read_file_txt('Csi',sDirectoryInput).reshape(nSectors, 1)
  #
    if nExecuta [nChoque] == 0:
        if nChoque == 0:
            VABrasil_pre, w_Brasil_pre, P_Brasil_pre, Y_pre, CrescTrab_pre, PBr_pre, xbilat_total_pre, GO_total_pre, p_total_pre, migr_pre = \
                Equilibrium(nCountries, nSectors, nTradebleSectors, D, nYears, beta, v, Pos, migracao, L, vfactor,
                            nMaxIter, nTolerance, Yinic, nAdjust, Csi, nChoque, sDirectoryInput, sDirectoryOutput)

            sFileName = "ResultadosSemChoque.xlsx"
            lSheet    = ['VABrasil', 'w_Brasil', 'P_Brasil', 'Y', 'cresc_trab', 'PBr', 'xbilat_total', 'GO_total',
                         'p_total', 'migr']
            lData     = [VABrasil_pre, w_Brasil_pre, P_Brasil_pre, Y_pre, CrescTrab_pre, PBr_pre, xbilat_total_pre,
                         GO_total_pre, p_total_pre, migr_pre]

        else:
            VABrasil_pos, w_Brasil_pos, P_Brasil_pos, Y_pos, CrescTrab_pos, PBr_pos, xbilat_total_pos, GO_total_pos, p_total_pos, migr_pos = \
                Equilibrium(nCountries, nSectors, nTradebleSectors, D, nYears, beta, v, Pos, migracao, L, vfactor,
                            nMaxIter, nTolerance, Yinic, nAdjust, Csi, nChoque, sDirectoryInput, sDirectoryOutput)

            sFileName = "ResultadosComChoque.xlsx"
            lSheet    = ['VABrasil_C', 'w_Brasil_C', 'P_Brasil_C', 'Y_C', 'cresc_trab_C', 'PBr_C', 'xbilat_total_C',
                         'GO_total_C', 'p_total_C', 'migr_C']
            lData     = [VABrasil_pos, w_Brasil_pos, P_Brasil_pos, Y_pos, CrescTrab_pos, PBr_pos, xbilat_total_pos,
                         GO_total_pos, p_total_pos, migr_pos]

        FuncoesApoio.write_data_excel(sDirectoryOutput, sFileName, lSheet, lData)
#        for each in range(len(lSheet)):
#            FuncoesApoio.write_file_excel(sDirectoryOutput ,lSheet[each]+".xlsx",lSheet[each], lData[each])
    else:
        if nChoque == 0:
            vSheet    = ['VABrasil', 'w_Brasil', 'P_Brasil', 'Y', 'cresc_trab', 'PBr', 'xbilat_total', 'GO_total',
                         'p_total', 'migr']
            VABrasil_pre, w_Brasil_pre, P_Brasil_pre, Y_pre, CrescTrab_pre, PBr_pre, xbilat_total_pre, GO_total_pre, \
            p_total_pre, migr_pre = FuncoesApoio.read_data_excel(sDirectoryOutput, "ResultadosSemChoque.xlsx", vSheet)
        else:
            vSheet    = ['VABrasil_C', 'w_Brasil_C', 'P_Brasil_C', 'Y_C', 'cresc_trab_C', 'PBr_C', 'xbilat_total_C',
                         'GO_total_C', 'p_total_C', 'migr_C']
            VABrasil_pos, w_Brasil_pos, P_Brasil_pos, Y_pos, CrescTrab_pos, PBr_pos, xbilat_total_pos, GO_total_pos, \
            p_total_pos, migr_pos = FuncoesApoio.read_data_excel(sDirectoryOutput, "ResultadosComChoque.xlsx", vSheet)



print("+++++++++++++++++++++++++++++++++++")
print("Tratamento de dados para resultados")
print("+++++++++++++++++++++++++++++++++++")

# ============================================================================================
# Tratamento dos dados para apresentação de resultados
# ============================================================================================
vDataSheet = []
vSheetName = []
# ============================================================================================
# Variação dos preços
# ============================================================================================
p_pre = np.zeros([nYears, nSectors], dtype=float)
p_pos = np.zeros([nYears, nSectors], dtype=float)
p_pre[0, :] = P_Brasil_pre[0, :]
p_pos[0, :] = P_Brasil_pos[0, :]
for t in range(1,nYears):
    for j in range(nSectors):
        p_pre[t, j] =P_Brasil_pre[t, j] * p_pre[t - 1, j]

        p_pos[t, j] =P_Brasil_pos[t, j] * p_pos[t - 1, j]

Crescimento_Preco = (p_pos / p_pre - 1) * 100
#
vDataSheet.append(Crescimento_Preco)
vSheetName.append('VariacaoPrecos')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Crescimento_Preco.xlsx', 'Sheet', Crescimento_Preco)
# ============================================================================================
# Variação dos Indice de Precos
# ============================================================================================
IP_pre = np.zeros([nYears], dtype=float)
IP_pos = np.zeros([nYears], dtype=float)

IP_pre[0] = PBr_pre[0]
IP_pos[0] = PBr_pos[0]

for t in range(1,nYears):
    IP_pre[t] = PBr_pre[t] * IP_pre[t - 1]
    IP_pos[t] = PBr_pos[t] * IP_pos[t - 1]
Crescimento_Indice_Preco = ((IP_pos / IP_pre - 1) * 100).reshape(nYears, 1)
#
vDataSheet.append(Crescimento_Indice_Preco)
vSheetName.append('IndicePrecos')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Crescimento_Indice_Preco.xlsx','Sheet' , Crescimento_Indice_Preco)
# ============================================================================================
# Variação dos salários
# ============================================================================================
S_pre = np.zeros([nYears, nSectors], dtype=float)
S_pos = np.zeros([nYears, nSectors], dtype=float)
S_pre[0, :] = w_Brasil_pre[0, :]
S_pos[0, :] = w_Brasil_pos[0, :]
for t in range(1,nYears):
    for j in range(nSectors):
        S_pre[t, j] = w_Brasil_pre[t, j] * S_pre[t - 1, j]
        S_pos[t, j] = w_Brasil_pos[t, j] * S_pos[t - 1, j]

IP_pre_aux = np.tile((IP_pre).reshape(nYears,1), nSectors)
S_pre = S_pre / IP_pre_aux
IP_pos_aux = np.tile((IP_pos).reshape(nYears,1), nSectors)
S_pos = S_pos / IP_pos_aux
Crescimento_Salario = (S_pos / S_pre - 1) * 100
#
vDataSheet.append(Crescimento_Salario)
vSheetName.append('VariacaoSalarios')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Crescimento_Salario.xlsx','Sheet' , Crescimento_Salario)
# ============================================================================================
# Variação do PIB (crescimento)
# ============================================================================================
VA_pre = VABrasil_pre / p_pre
VA_pos = VABrasil_pos / p_pos

VA_total_pre = (sum(VA_pre.T)).reshape(nYears,1)
VA_total_pos = (sum(VA_pos.T)).reshape(nYears,1)


VA_pre = np.hstack((VA_pre, VA_total_pre))
VA_pos = np.hstack((VA_pos, VA_total_pos))

Crescimento = (VA_pos / VA_pre - 1) * 100
#
vDataSheet.append(Crescimento)
vSheetName.append('VariacaoPIB')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Crescimento.xlsx','Sheet' , Crescimento)
# ============================================================================================
# Variação do PIB setoriais (crescimento setoriais)
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

VA_setores_pre = np.concatenate(((VA_agricultura_pre).reshape(nYears,1), (VA_ind_extr_pre).reshape(nYears,1), (VA_ind_tran_pre).reshape(nYears,1), (VA_serv_pre).reshape(nYears,1)),axis=1)
VA_setores_pos = np.concatenate(((VA_agricultura_pos).reshape(nYears,1), (VA_ind_extr_pos).reshape(nYears,1), (VA_ind_tran_pos).reshape(nYears,1), (VA_serv_pos).reshape(nYears,1)),axis=1)
Crescimento_setores = (VA_setores_pos / VA_setores_pre - 1) * 100
#
vDataSheet.append(Crescimento_setores)
vSheetName.append('VariacaoPIBSetorial')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Crescimento_setores.xlsx','Sheet' , Crescimento_setores)

# ============================================================================================
# Variação do Cambio
# ============================================================================================
GO_pre_time = np.zeros([nSectors, nCountries], dtype=float)
GO_pos_time = np.zeros([nSectors, nCountries], dtype=float)
p_pre_time = np.zeros([nSectors, nCountries], dtype=float)
p_pos_time = np.zeros([nSectors, nCountries], dtype=float)
tau_pre_time = np.zeros([nSectors * nCountries, nCountries], dtype=float)
tau_pos_time = np.zeros([nSectors * nCountries, nCountries], dtype=float)

lRead = ['Tarifas', 'TarifasZero']
Tarifas, TarifasZero = FuncoesApoio.read_data_txt(lRead, sDirectoryInput)
tau = np.vstack((1 + Tarifas / 100, np.ones([15 * nCountries, nCountries], dtype=float)))
tau_pre = np.tile(tau, (nYears, 1))
tau = np.vstack((1 + TarifasZero / 100, np.ones([15 * nCountries, nCountries], dtype=float)))
tau_pos = np.tile(tau, (nYears, 1))

GO_totalaux_pre = np.zeros([nSectors, nCountries], dtype=float)
GO_totalaux_pos = np.zeros([nSectors, nCountries], dtype=float)
tau_pre_aux = np.zeros([nCountries * nTradebleSectors, nCountries], dtype=float)
tau_pos_aux = np.zeros([nCountries * nTradebleSectors, nCountries], dtype=float)
total_comex_pre = np.zeros([nCountries, nCountries], dtype=float)
total_comex_pos = np.zeros([nCountries, nCountries], dtype=float)
Cambio = np.zeros([nYears, nCountries], dtype=float)

for t in range(nYears):
    for j in range(nSectors):
        for n in range(nCountries):
            GO_pre_time[j, n] = GO_total_pre[t * nSectors + j, n]
            GO_pos_time[j, n] = GO_total_pos[t * nSectors + j, n]
            p_pre_time[j, n] = p_total_pre[t * nSectors + j, n]
            p_pos_time[j, n] = p_total_pos[t * nSectors + j, n]

    export_pre = np.zeros([nSectors * nCountries, nCountries], dtype=float)
    export_pos = np.zeros([nSectors * nCountries, nCountries], dtype=float)
    for i in range(nSectors * nCountries):
        for n in range(nCountries):
            export_pre[i, n] = xbilat_total_pre[t * nSectors * nCountries + i, n]
            export_pos[i, n] = xbilat_total_pos[t * nSectors * nCountries + i, n]
            tau_pre_time[i, n] = tau_pre[t * nSectors * nCountries + i, n]
            tau_pos_time[i, n] = tau_pos[t * nSectors * nCountries + i, n]

    GO_aux_pre = sum(GO_pre_time)
    GO_aux_pos = sum(GO_pos_time)

    for j in range(nSectors):
        for n in range(nCountries):
            GO_totalaux_pre[j, n] = GO_aux_pre[n]
            GO_totalaux_pos[j, n] = GO_aux_pos[n]

    GO_pesos_pre = GO_pre_time / GO_totalaux_pre
    GO_pesos_pos = GO_pos_time / GO_totalaux_pos

    ind_p_pre = np.prod(p_pre_time ** (GO_pesos_pre), axis=0)
    ind_p_pos = np.prod(p_pos_time ** (GO_pesos_pos), axis=0)

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
            # Exports
            export_pre_aux[m, n] = sum(export_pre[m: M * nTradebleSectors:M, n]).T
            # Exports
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

    ind_p_pre_aux = np.tile((ind_p_pre.T).reshape(nCountries,1), (1, nCountries))
    ind_p_pos_aux = np.tile((ind_p_pos.T).reshape(nCountries,1), (1, nCountries))

    p_exterior_pre = np.prod(ind_p_pre_aux ** (pesos_comex_pre), axis=0)
    p_exterior_pos = np.prod(ind_p_pos_aux ** (pesos_comex_pos), axis=0)

    cambio_pre = p_exterior_pre / ind_p_pre
    cambio_pos = p_exterior_pos / ind_p_pos

    for n in range(nCountries):
        Cambio[t, n] = cambio_pos[n] / cambio_pre[n]

Crescimento_cambio = np.zeros([nYears, nCountries], dtype=float)
Crescimento_cambio[0, :] = Cambio[0, :]
for t in range(1, nYears):
    Crescimento_cambio[t, :] = Cambio[t, :] * Crescimento_cambio[t - 1, :]
#
vDataSheet.append(Crescimento_cambio)
vSheetName.append('VariacaoCambial')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Crescimento_cambio.xlsx', 'Sheet' ,Crescimento_cambio)
# ============================================================================================
# Variação das exportações país x setor
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

for t in range(nYears):
    export_pre = np.zeros([nSectors * nCountries, nCountries], dtype=float)
    export_pos = np.zeros([nSectors * nCountries, nCountries], dtype=float)
    for i in range(nSectors * nCountries):
        for n in range(nCountries):
            export_pre[i, n] = xbilat_total_pre[t * nSectors * nCountries + i, n]
            export_pos[i, n] = xbilat_total_pos[t * nSectors * nCountries + i, n]
            tau_pre_time[i, n] = tau_pre[t * nSectors * nCountries + i, n]
            tau_pos_time[i, n] = tau_pos[t * nSectors * nCountries + i, n]

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

            cresc_export[t * nSectors + j, n] = export_pos_aux[j, n] / export_pre_aux[j, n]

            export_brasil_pre_aux[t * nSectors + j, n] = export_pre_aux[j, n]
            export_brasil_pos_aux[t * nSectors + j, n] = export_pos_aux[j, n]

    export_pre_soma = sum(export_pre_aux)
    export_pos_soma = sum(export_pos_aux)
    for n in range(nCountries):
        cresc_export_total[t, n] = export_pos_soma[n] / export_pre_soma[n]

for t in range(nYears):
    for i in range (nTradebleSectors):
        cresc_export_brasil[t,i] = cresc_export[t*nSectors+i,Pos]
        export_brasil_pre[t,i] = export_brasil_pre_aux[t*nSectors+i,Pos]
        export_brasil_pos[t,i] = export_brasil_pos_aux[t*nSectors+i,Pos]

#
vDataSheet.append(cresc_export_total)
vSheetName.append('CrescExportTotal')
#
vDataSheet.append(cresc_export_brasil)
vSheetName.append('CrescExportBrasil')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Crescimento_export.xlsx','Sheet',cresc_export_total)
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Export_Brasil.xlsx','Sheet',cresc_export_brasil)

# ============================================================================================
# Variação das exportações setoriais
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
#
vDataSheet.append(Crescimento_export_setores)
vSheetName.append('ExportacaoSetorial')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Crescimento_export_setores.xlsx','Sheet',Crescimento_export_setores)
# ============================================================================================
# Variação das exportações país x país
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
        tau_pre_time[i, n] = tau_pre[(nYears-2) * nSectors * nCountries + i, n]
        tau_pos_time[i, n] = tau_pos[(nYears-2) * nSectors * nCountries + i, n]

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
#
vDataSheet.append(Crescimento_export_por_pais)
vSheetName.append('ExportacaoPorPais')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Crescimento_export_por_pais.xlsx', 'Sheet' ,Crescimento_export_por_pais)

# ============================================================================================
# Variação das exportações  Brasil x país x setor
# ============================================================================================
export_por_produto_pais_pos = (export_pos / export_pre - 1)*100
export_por_produto_pais_pos[np.isnan(export_por_produto_pais_pos)]=0
export_Brasil_pos = export_por_produto_pais_pos[:, Pos]
export_Brasil_final = np.zeros([nTradebleSectors, nCountries], dtype=float)

for n in range(nCountries):
    export_Brasil_final[:,n] = export_Brasil_pos[n:nCountries*nTradebleSectors:nCountries]
#
vDataSheet.append(export_Brasil_final)
vSheetName.append('ExportacaoBrasil')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'export_Brasil_final.xlsx', 'Sheet' ,export_Brasil_final)
# ============================================================================================
# Variação das importações  Brasil x país x setor
# ============================================================================================

#
#Essa operação abaixo, por causa das importações de coque
#for i = 1:1881
#    for n=1:33
#    if export_pre(i,n) < 1e-007 & export_pre(i,n)>0
#    indice_import_Brasil(i,n) = 1;
#    else indice_import_Brasil(i,n) = 0;
#    end
#    end
#end

#for n = 1:nCountries
#    indice_import_Brasil_aux(:,n) = indice_import_Brasil(4:nCountries:nCountries*42,n);
#end
#%}
import_Brasil_final = np.zeros([nTradebleSectors, nCountries], dtype=float)
for n in range(nCountries):
    import_Brasil_final[:,n] = export_por_produto_pais_pos[Pos:nCountries*nTradebleSectors:nCountries,n]
#
vDataSheet.append(import_Brasil_final)
vSheetName.append('ImportacaoBrasil')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'import_Brasil_final.xlsx', 'Sheet' ,import_Brasil_final)
# ============================================================================================
# Variação do Mercado de trabalho
# ============================================================================================
num_trab_pre = np.zeros([nYears, D], dtype=float)
num_trab_pos = np.zeros([nYears, D], dtype=float)

num_trab_pre[0,:] = L * CrescTrab_pre[0,:]
num_trab_pos[0,:] = L * CrescTrab_pos[0,:]

for t in range(1,nYears):
    num_trab_pre[t, :] = num_trab_pre[t-1,:] * CrescTrab_pre[t, :]
    num_trab_pos[t, :] = num_trab_pos[t-1,:] * CrescTrab_pos[t, :]

Cresc_PO = (num_trab_pos / num_trab_pre-1)*100
#
vDataSheet.append(Cresc_PO)
vSheetName.append('VariacaoPO')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Cresc_PO.xlsx', 'Sheet' ,Cresc_PO)


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
#
vDataSheet.append(Crescimento_PO_setores)
vSheetName.append('POSetorial')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Crescimento_PO_setores.xlsx', 'Sheet' ,Crescimento_PO_setores)
# ============================================================================================
# Variação da Produtividade
# ============================================================================================
num_trab_aux2_pre = np.sum(num_trab_aux_pre,axis=1)
num_trab_aux2_pre = np.concatenate((num_trab_aux_pre, num_trab_aux2_pre.reshape(nYears,1)), axis=1)
num_trab_aux2_pos = np.sum(num_trab_aux_pos,axis=1)
num_trab_aux2_pos = np.concatenate((num_trab_aux_pos, num_trab_aux2_pos.reshape(nYears,1)), axis=1)

produtividade_pre = VA_pre / num_trab_aux2_pre
produtividade_pos = VA_pos / num_trab_aux2_pos

Crescimento_produtividade = (produtividade_pos/produtividade_pre-1)*100
#
vDataSheet.append(Crescimento_produtividade)
vSheetName.append('Produtividade')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Crescimento_produtividade.xlsx', 'Sheet' ,Crescimento_produtividade)

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
#
vDataSheet.append(Cresc_produtividade_setores)
vSheetName.append('ProdutividadeSetorial')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Cresc_produtividade_setores.xlsx', 'Sheet' ,Cresc_produtividade_setores)


# ============================================================================================
# Variação dos CUSTOS DE AJUSTE
# ============================================================================================


# ============================================================================================
# Variação dos Salários setoriais
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
#
vDataSheet.append(Crescimento_Salario_setores)
vSheetName.append('SalariosSetoriais')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Crescimento_Salario_setores.xlsx', 'Sheet' ,Crescimento_Salario_setores)
# ============================================================================================
# Variação dos Preços setoriais
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

p_pre[0,:] = P_Brasil_pre[0,:]
p_pos[0,:] = P_Brasil_pos[0,:]
for t in range(1,nYears):
    for j in  range(nSectors):
        p_pre[t, j] = P_Brasil_pre[t, j] * p_pre[t - 1, j]
        p_pos[t, j] = P_Brasil_pos[t, j] * p_pos[t - 1, j]

alphas = FuncoesApoio.read_file_txt('Alphas',sDirectoryInput)
#alphas = np.ones([nSectors,nCountries], dtype=float)
alphas_aux = (alphas.T).reshape(nCountries, nSectors)

for t in range(nYears):
    P_agricultura_pre[t] = np.prod(p_pre[t, 0:14] ** (alphas_aux[Pos, 0:14]), axis=0)
    P_agricultura_pos[t] = np.prod(p_pos[t, 0:14]  ** (alphas_aux[Pos, 0:14]), axis=0)
    P_ind_extr_pre[t]    = np.prod(p_pre[t, 14:18] ** (alphas_aux[Pos, 14:18]), axis=0)
    P_ind_extr_pos[t]    = np.prod(p_pos[t, 14:18] ** (alphas_aux[Pos, 14:18]), axis=0)

    P_ind_tran_pre[t]    = np.prod(p_pre[t, 18:42] ** (alphas_aux[Pos, 18:42]), axis=0)
    P_ind_tran_pos[t]    = np.prod(p_pos[t, 18:42] ** (alphas_aux[Pos, 18:42]), axis=0)
    P_serv_pre[t]        = np.prod(p_pre[t, 42:57] ** (alphas_aux[Pos, 42:57]), axis=0)
    P_serv_pos[t]        = np.prod(p_pos[t, 42:57] ** (alphas_aux[Pos, 42:57]), axis=0)

Precos_setores_pre = np.concatenate((P_agricultura_pre.reshape(nYears,1), P_ind_extr_pre.reshape(nYears,1), P_ind_tran_pre.reshape(nYears,1), P_serv_pre.reshape(nYears,1)), axis=1)
Precos_setores_pos = np.concatenate((P_agricultura_pos.reshape(nYears,1), P_ind_extr_pos.reshape(nYears,1), P_ind_tran_pos.reshape(nYears,1), P_serv_pos.reshape(nYears,1)), axis=1)

Crescimento_Precos_setores = (Precos_setores_pos / Precos_setores_pre-1)*100
#
vDataSheet.append(Crescimento_Precos_setores)
vSheetName.append('PrecosSetoriais')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Crescimento_Precos_setores.xls', 'Sheet' ,Crescimento_Precos_setores)

# ============================================================================================
# Variação da Taxa de ajuste - PO
# ============================================================================================
num_trab_pre    = np.zeros([nYears, D], dtype=float)
num_trab_pos    = np.zeros([nYears, D], dtype=float)
cum_ajuste_aux  = np.zeros([nYears, D], dtype=float)
ajuste_PO       = np.zeros([nYears], dtype=float)
num_trab_pre[0,:] = L * CrescTrab_pre[0,:]
num_trab_pos[0,:] = L * CrescTrab_pos[0,:]

for t in range(1,nYears):
    num_trab_pre[t,:] = num_trab_pre[t-1,:] * CrescTrab_pre[t,:]
    num_trab_pos[t,:] = num_trab_pos[t-1,:] * CrescTrab_pos[t,:]
for t in range(nYears):
    cum_ajuste_aux[t,:] = .5 * abs(num_trab_pos[t,:]-num_trab_pre[t,:])

cum_ajuste = sum(cum_ajuste_aux,1)

for t in range(nYears):
    ajuste_PO[t] = 100*(cum_ajuste[t]/cum_ajuste[nYears-1])
#
vDataSheet.append(ajuste_PO)
vSheetName.append('AjustePO')
#FuncoesApoio.write_file_excel(sDirectoryOutput ,'Ajuste_PO.xlsx', 'Sheet' ,ajuste_PO)
# ============================================================================================
# Variação das exportações setoriais só para o pos
# ============================================================================================


# ============================================================================================
# Variação das importações setoriais só do pos
# ============================================================================================


# ============================================================================================
# Gravacao ResultadoModelo
# ============================================================================================
FuncoesApoio.write_data_excel(sDirectoryOutput, "ResultadoModelo.xlsx", vSheetName, vDataSheet)

print("End")
sys.exit(0)