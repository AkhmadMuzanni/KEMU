# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 22:05:22 2018

@author: USER
"""
import numpy as np
import csv

#dataClass = [[[1.0,1.0,0.0,0.0],0],[[0.0,0.0,0.0,1.0],1]]
#dataTraining = [[[0.0,0.0,1.0,1.0],1],[[1.0,0.0,0.0,0.0],0],[[0.0,1.0,1.0,0.0],1]]

def read_csv(file_name):
    array_2D = []
    with open(file_name, 'rb') as csvfile:
        read = csv.reader(csvfile, delimiter=';')
        for row in read:
            array_2D.append(row)
    return array_2D

data1 = read_csv('data/dataColor.csv')
data2 = read_csv('data/dataTrainColor.csv')
data3 = read_csv('data/dataTestColor.csv')
dataTrain = ((np.array(data1[:]))[:,1:-1]).astype(np.float16).tolist()
dataC = ((np.array(data2[:]))[:,1:-1]).astype(np.float16).tolist()
dataTesting = ((np.array(data3[:]))[:,1:-1]).astype(np.float16).tolist()
classDataTrain = ((np.array(data1[:]))[:,-1:]).astype(int).tolist()
classDataClass = ((np.array(data2[:]))[:,-1:]).astype(int).tolist()
dataTraining = []
dataClass = []
#dataTesting = []
for i in range(len(dataTrain)):
    dataArray = []
    dataArray.append(dataTrain[i])
    dataArray.append(classDataTrain[i][0])
    dataTraining.append(dataArray)

for i in range(len(dataC)):
    dataArray2 = []
    dataArray2.append(dataC[i])
    dataArray2.append(classDataClass[i][0])
    dataClass.append(dataArray2)
    


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

def norm(listElement):
    normResult = listElement[:]
    sumValue = sum(listElement)
    for i in range(len(normResult)):
        normResult[i] = np.divide(listElement[i],sumValue)
    return normResult

def getWN(diff):
    return norm(diff)

def threshold(i):
    if (i < 0.0001):
        return 0.0001
    elif (i > 1):
        return 1
    else:
        return i
    

countLVQ1 = 0
countLVQ2 = 0
countLVQ21 = 0
countLVQ3 = 0

wFeature = [1]*len(dataClass[0][0])
wF = [1]*len(dataClass[0][0])

# ITERATION LVQ
iterasi = 50
for x in range(iterasi):
    
    # Find Manhattan Distance
    jarak = np.zeros((len(dataClass),len(dataTraining)),dtype=np.float64)
    for i in range(len(dataTraining)):
        for k in range(len(dataClass)):
            jarak[k][i] = 0
            for j in range(len(dataTraining[i][0])):
                jarak[k][i] += wFeature[j] * np.abs(dataTraining[i][0][j] - weightMatrix[k][j])

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
            t = dataTraining[i][1]
            yc1 = int(jMatrix[i][0][1]) # The nearest class from the data training 
            yc2 = int(jMatrix[i][1][1]) # The second nearest class from the data training 
            dc1 = jMatrix[i][0][0] # The distance of the nearest class from the data training 
            dc2 = jMatrix[i][1][0] # The distance of the second nearest class from the data training 
            
            di = [np.abs(a-b) for a,b in zip(dataTraining[i][0],dataClass[yc1][0])]
            dj = [np.abs(a-b) for a,b in zip(dataTraining[i][0],dataClass[yc2][0])]
            wn = getWN([a-b for a,b in zip(di,dj)])            
            
            
            for k in range(len(weightMatrix[j])):
                
                
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
                
                #if ( (min(np.divide(dc1,dc2),np.divide(dc2,dc1)) > ((1-epsilon2)*(1+epsilon2))) and ((yc1 == t and yc2 != t) or (yc1 != t and yc2 == t)) ):
                if ( ((yc1 == t and yc2 != t) or (yc1 != t and yc2 == t)) ):
                    wFeature = norm([threshold( (1-alpha)*z + alpha*zw ) for z,zw in zip(wF,wn)])
    alpha = 0.8*alpha
            
# Validation
#dataTest = [0.0,0.0,1.0,1.0]
#dataTest = [1.0,0.0,0.0,0.0]
#dataTest = [0.0,1.0,1.0,0.0]
dataTest = dataTesting[2]
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

#res = [np.abs(a-b) for a,b in zip(dataClass[0][0], dataClass[1][0])]
#print(norm(res))
    


