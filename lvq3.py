# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 22:05:22 2018

@author: USER
"""
import numpy as np

dataClass = [[[1.0,1.0,0.0,0.0],0],[[0.0,0.0,0.0,1.0],1]]
dataTraining = [[[0.0,0.0,1.0,1.0],1],[[1.0,0.0,0.0,0.0],0],[[0.0,1.0,1.0,0.0],1]]

alpha = 0.1
beta = 0.1*alpha
epsilon= 0.35
epsilon2= 0.2

# Initialize Weight Matrix
weightMatrix = np.zeros((len(dataClass),len(dataTraining[0][0])),dtype=np.float64)
for i in range(len(weightMatrix)):
    for j in range(len(dataTraining[i][0])):
        weightMatrix[i][j] = dataClass[i][0][j]

def takeFirst(elem):
    return elem[0]

countLVQ1 = 0
countLVQ2 = 0
countLVQ21 = 0
countLVQ3 = 0
# ITERATION LVQ
iterasi = 50
for x in range(iterasi):
    
    # Find Manhattan Distance
    jarak = np.zeros((len(dataClass),len(dataTraining)),dtype=np.float64)
    for i in range(len(dataTraining)):
        for k in range(len(dataClass)):
            jarak[k][i] = 0
            for j in range(len(dataTraining[i][0])):
                jarak[k][i] += np.abs(dataTraining[i][0][j] - weightMatrix[k][j])

    jMatrix = [[]] * len(dataTraining)
    for i in range(len(dataTraining)):
        jVal = []
        for j in range(len(dataClass)):
            jVal.append([jarak[j][i],j])
        jMatrix[i] = jVal
        jMatrix[i].sort(key=takeFirst)
    
    '''
    # Initialize jValue
    jValue = np.zeros((len(dataTraining)),dtype=int)
    for i in range(len(jValue)):
        winnerClass = dataClass[0][1]
        minValue = 999999
        for j in range(len(jarak)):
            if(jarak[j][i] < minValue):
                minValue = jarak[j][i]
                winnerClass = j
        jValue[i] = winnerClass
    '''
    
    for i in range(len(dataTraining)):
        for j in range(len(weightMatrix)):
            for k in range(len(weightMatrix[j])):
                t = dataTraining[i][1]
                yc1 = int(jMatrix[i][0][1]) # The nearest class from the data training 
                yc2 = int(jMatrix[i][1][1]) # The second nearest class from the data training 
                dc1 = jMatrix[i][0][0] # The distance of the nearest class from the data training 
                dc2 = jMatrix[i][1][0] # The distance of the second nearest class from the data training 
                
                if ( (min(np.divide(dc1,dc2),np.divide(dc2,dc1)) > ((1-epsilon2)*(1+epsilon2))) and (yc1 == t and yc2 == t) ):
                    countLVQ3 += 1
                    if(j == yc1 or j == yc2):
                        weightMatrix[j][k] = (1-beta)*weightMatrix[j][k] + beta*dataTraining[i][0][k]
                                    
                elif ( (min(np.divide(dc1,dc2),np.divide(dc2,dc1)) > (1-epsilon)) and (max(np.divide(dc1,dc2),np.divide(dc2,dc1)) < (1+epsilon)) and ((yc1 == t and yc2 != t) or (yc1 != t and yc2 == t)) ):
                    countLVQ21 += 1
                    if(j == yc1):
                        if(t == yc1):
                            weightMatrix[j][k] = (1-alpha)*weightMatrix[j][k] + alpha*dataTraining[i][0][k]
                        else:
                            weightMatrix[j][k] = (1+alpha)*weightMatrix[j][k] - alpha*dataTraining[i][0][k]
                    elif(j == yc2):
                        if(t == yc2):
                            weightMatrix[j][k] = (1-alpha)*weightMatrix[j][k] + alpha*dataTraining[i][0][k]
                        else:
                            weightMatrix[j][k] = (1+alpha)*weightMatrix[j][k] - alpha*dataTraining[i][0][k]
                            
                elif ((np.divide(jMatrix[i][0][0],jMatrix[i][1][0]) > (1 - epsilon)) and (np.divide(jMatrix[i][1][0],jMatrix[i][0][0]) < (1 + epsilon)) and (jMatrix[i][1][1] == dataTraining[i][1])):
                    countLVQ2 += 1
                    if(j == jMatrix[i][0][1]):
                        weightMatrix[j][k] = (1+alpha)*weightMatrix[j][k] + alpha*dataTraining[i][0][k]
                    elif(j == jMatrix[i][1][1]):
                        weightMatrix[j][k] = (1-alpha)*weightMatrix[j][k] + alpha*dataTraining[i][0][k]
                        
                elif (j == jMatrix[i][0][1]):
                    countLVQ1 += 1
                    if(j == dataTraining[i][1]):
                        weightMatrix[j][k] = (1-alpha)*weightMatrix[j][k] + alpha*dataTraining[i][0][k]
                    else:
                        weightMatrix[j][k] = (1+alpha)*weightMatrix[j][k] - alpha*dataTraining[i][0][k]
    alpha = 0.8*alpha
            
# Validation
#dataTest = [0.0,0.0,1.0,1.0]
#dataTest = [1.0,0.0,0.0,0.0]
dataTest = [0.0,1.0,1.0,0.0]
classResult = 0
minValue = 99999
for i in range(len(weightMatrix)):
    sumValue = 0
    for j in range(len(weightMatrix[i])):
        sumValue += np.power(dataTest[j] - weightMatrix[i][j],2)
    if (sumValue < minValue):
        minValue = sumValue
        classResult = i
print('Class Result = '+str(classResult))
print(countLVQ1/4)
print(countLVQ2/4)
print(countLVQ21/4)
print(countLVQ3/4)
    

