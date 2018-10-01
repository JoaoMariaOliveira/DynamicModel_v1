# ============================================================================================
# Support Functions of Input/output
# ============================================================================================

import numpy as np
import pandas as pd
import xlrd
import numpy.core.multiarray
from pandas import ExcelWriter
from pandas import ExcelFile
# ============================================================================================
# Função que lê dados de um arquivo txt e retorna uma matriz
# lê de uma pasta input e pode ler vários ao mesmo tempo
# ============================================================================================


def read_data_txt(lDataToRead, sDirectory):
    lDataContainer = []
    for each in lDataToRead:
        temp = np.loadtxt(sDirectory+each+'.txt')
        lDataContainer.append(temp)
    return lDataContainer
# ============================================================================================
# Função que lê dados de um arquivo txt e retorna uma matriz
# lê de uma pasta input
# ============================================================================================


def read_file_txt(sFileName, sDirectory):
    sFileName = sDirectory + sFileName + ".txt"
    temp: numpy.core.multiarray.ndarray = np.loadtxt(sFileName, dtype=np.float64)
    return temp
# ============================================================================================
# Função que lê dados de um arquivo csv e retorna uma matriz
# lê de uma pasta input se não for informada outra
# ============================================================================================


def read_file_csv(sFileName, sDirectory):
    sFileName = sDirectory + sFileName
    temp: numpy.core.multiarray.ndarray = np.genfromtxt(sFileName+'.csv', delimiter=',')
    return temp
# ============================================================================================
# Função que lê dados de um arquivo csv e retorna uma matriz
# lê de uma pasta input e pode ler vários ao mesmo tempo
# ============================================================================================


def read_data_csv(lDataToRead, sDirectory):
    lDataContainer = []
    for each in lDataToRead:
        temp: numpy.core.multiarray.ndarray=np.genfromtxt(sDirectory+each+'.csv', delimiter=',')
        lDataContainer.append(temp)
    return lDataContainer
# ============================================================================================
# Função que grava dados em um arquivo csv
# Grava em uma pasta output e pode gravar vários ao mesmo tempo
# ============================================================================================


def write_data_csv(lDataToWrite, lData, sDirectory):
    for each in range(len(lDataToWrite)):
        np.savetxt(sDirectory + lDataToWrite[each] + '.csv', lData[each], delimiter=',')
# ============================================================================================
# Função que grava dados em um arquivo csv
# Grava em uma pasta output
# ============================================================================================


def write_file_csv(data_to_save, Data):
    np.savetxt('./Output/' + data_to_save + '.csv', Data, delimiter=',')

# ============================================================================================
# Função que grava dados em um arquivo excel
# Grava em uma pasta output e pode gravar várias planilhas em um mesmo arquivo
# ============================================================================================
def write_data_excel(sDirectory, FileName, lSheetName, lDataSheet):
    Writer = pd.ExcelWriter(sDirectory + FileName, engine='xlsxwriter')
    df=[]
    for each in range(len(lSheetName)):
        df.append(pd.DataFrame(lDataSheet[each], dtype=float))
        df[each].to_excel(Writer, lSheetName[each], header=False, index=False)
    Writer.save()

# ============================================================================================
# Função que grava dados em um arquivo excel
# Grava em uma pasta output e pode gravar uma só planilha em um mesmo arquivo
# ============================================================================================

def write_file_excel(sDirectory, sFileName, sSheet, mData):
    Writer = pd.ExcelWriter(sDirectory + sFileName, engine='xlsxwriter')
    df = pd.DataFrame(mData, dtype=float)
    df.to_excel(Writer, sheet_name=sSheet, header=False, index=False)
    Writer.save()

# ============================================================================================
# Função que lê dados de um arquivo excel
# lê de uma pasta (default input) e pode ler várias planilhas em um mesmo arquivo
# ============================================================================================

def read_data_excel(sDirectory, sFileName, vSheet):
    data = []
    for each in range(len(vSheet)):
        df = pd.read_excel(sDirectory + sFileName, sheet_name=vSheet[each], header=None, index_col=None)
        data.append(df.values)

    return data
# ============================================================================================
# Função que lê dados de um arquivo excel
# lê de uma pasta (default input) e pode ler uma planilhas arquivo
# ============================================================================================


def read_file_excel(vSheet,sDirectory):
    if sDirectory == None:
        sDirectory = './Input/'
    data=[]
    for each  in range(len(vSheet)):
        sFileName = sDirectory + vSheet[each]+".xlsx"
        df = pd.read_excel(sFileName, sheet_name=vSheet[each], header=None, index_col=None)
        data.append(df.values)

    return data

