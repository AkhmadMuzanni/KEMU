# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 23:15:48 2018

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

data = read_csv('data/dataTrain1.csv')
dataTrain = ((np.array(data[:]))[:,1:-1]).astype(np.float16).tolist()
classDataTrain = ((np.array(data[:]))[:,-1:]).astype(int).tolist()
dataTrainFix = []
for i in range(len(dataTrain)):
    dataArray = []
    dataArray.append(dataTrain[i])
    dataArray.append(classDataTrain[i][0])
    dataTrainFix.append(dataArray)
