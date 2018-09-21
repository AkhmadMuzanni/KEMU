# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 22:05:22 2018

@author: USER
"""
import numpy as np

dataClass = [[[1.0,1.0,0.0,0.0],1],[[0.0,0.0,0.0,1.0],2]]
dataTraining = [[[0.0,0.0,1.0,1.0],2],[[1.0,0.0,0.0,0.0],1],[[0.0,1.0,1.0,0.0],2]]

jarak = np.zeros((len(dataClass),len(dataTraining)),dtype=np.float64)
for i in range(len(dataTraining)):
    for k in range(len(dataClass)):
        jarak[k][i] = 0
        for j in range(len(dataTraining[i][0])):
            jarak[k][i] += np.abs(dataTraining[i][0][j] - dataClass[k][0][j])

print(dataClass)
print(dataTraining)