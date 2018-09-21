# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 22:05:22 2018

@author: USER
"""
import numpy as np

dataClass = [[[1.0,1.0,0.0,0.0],0],[[0.0,0.0,0.0,1.0],1]]
dataTraining = [[[0.0,0.0,1.0,1.0],1],[[1.0,0.0,0.0,0.0],0],[[0.0,1.0,1.0,0.0],1]]

alpha = 0.1

# Initialize Weight Matrix
weightMatrix = np.zeros((len(dataClass),len(dataTraining[0][0])),dtype=np.float64)
for i in range(len(weightMatrix)):
    for j in range(len(dataTraining[i][0])):
        weightMatrix[i][j] = dataClass[i][0][j]



iterasi = 15
for x in range(iterasi):
    
    # Find Manhattan Distance
    jarak = np.zeros((len(dataClass),len(dataTraining)),dtype=np.float64)
    for i in range(len(dataTraining)):
        for k in range(len(dataClass)):
            jarak[k][i] = 0
            for j in range(len(dataTraining[i][0])):
                jarak[k][i] += np.abs(dataTraining[i][0][j] - weightMatrix[k][j])
    
    # Initialize jValue
    jValue = np.zeros((len(dataTraining)),dtype=int)
    for i in range(len(jValue)):    
        minClass = dataClass[0][1]
        minValue = 999999
        for j in range(len(jarak)):
            if(jarak[j][i] < minValue):
                minValue = jarak[j][i]
                minClass = j
        jValue[i] = minClass    
    
    for i in range(len(dataTraining)):
        for j in range(len(weightMatrix)):
            for k in range(len(weightMatrix[j])):
                if (j == jValue[i]):
                    if(j == dataTraining[i][1]):
                        #print("Masuk")
                        weightMatrix[j][k] = (1-alpha)*weightMatrix[j][k] + alpha*dataTraining[i][0][k]
                    else:                        
                        weightMatrix[j][k] = (1+alpha)*weightMatrix[j][k] - alpha*dataTraining[i][0][k]
    alpha = 0.8*alpha
            

print(dataClass)
print(dataTraining)