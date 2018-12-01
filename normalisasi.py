# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 22:16:21 2018

@author: USER
"""
import csv
import numpy as np

def read_csv(file_name):
    array_2D = []
    with open(file_name, 'rb') as csvfile:
        read = csv.reader(csvfile, delimiter=';')
        for row in read:
            array_2D.append(row)
    return array_2D

dataAsli = read_csv('data/dataNormal1.csv')
dataTrain = ((np.array(dataAsli[:]))[:,1:-1]).astype(np.float64).tolist()
maxValue = [-999999]*len(dataTrain[0])
minValue = [999999]*len(dataTrain[0])
maxVal = -999999
minVal = 999999
for i in range(len(dataTrain)):
    for j in range(len(dataTrain[i])):
        if (dataTrain[i][j] > maxValue[j]):
            maxValue[j] = dataTrain[i][j]
        if (dataTrain[i][j] < minValue[j]):
            minValue[j] = dataTrain[i][j]

dataNorm = np.copy(dataTrain).tolist()
for i in range(len(dataNorm)):
    for j in range(len(dataNorm[i])):
        dataNorm[i][j] = (dataNorm[i][j] - minValue[j]) / (maxValue[j] - minValue[j])
        