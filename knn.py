# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:10:13 2018

@author: USER
"""
import csv
import os
import numpy as np

def read_csv(file_name):
    array_2D = []
    with open(file_name, 'rb') as csvfile:
        read = csv.reader(csvfile, delimiter=';')
        for row in read:
            array_2D.append(row)
    return array_2D

def buildFold(foldTest):
    path = "data/realTraining/fold/"
    #pathOutput = "D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\DATA01\\"
    tot = os.listdir(path)    
    #kelas = 1
    fold = [] 
    for filename in tot:       
        fold.append(read_csv(path+filename)) # Fold 1
    
    #foldTest = [0]
    dataUji = []
    for i in range(len(foldTest)):
        dataUji.extend(fold[foldTest[i]])
    dataLatih = []
    for i in range(len(fold)):
        if (i not in foldTest):
            dataLatih.extend(fold[i])
    return dataLatih, dataUji

def featureSelection(listFeatures):
    res = np.transpose(listFeatures)
    result = []
    #eliminated = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40] # Warna
    #eliminated = [41,42,43,44,45,46,47,48,49,50,51,52] # Haralick
    
    #eliminated = [4,5,14,15,24,25,26,34,35] # Relief 
    #eliminated = [1,2,9,10,11,12,19,20,21,22,29,30,31,32,39,40,41,42,44,45,46,48,49] # Korelasi
    #eliminated = [1,2,3,5,6,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,43,46,47,48,49,50,51,52] # CFS
    eliminated = [] # Tanpa Seleksi
    for i in range(len(res)):
        if ((i+1) not in eliminated):
            resJ = []
            for j in range(len(res[i])):
                resJ.append(res[i][j])
            result.append(resJ)
    return np.transpose(result).tolist()

#data1 = read_csv('data/realTraining/dataTrain80204.csv') # Data Training
#data3 = read_csv('data/realTraining/dataTest80204.csv') # Data Testing
data2 = read_csv('data/realTraining/dataClass80204.csv') # Data Class (Vector Reference)
testFold = [0]
fold = 'fold0'
data1, data3 = buildFold(testFold)

dataTrain = ((np.array(data1[:]))[:,1:-1]).astype(np.float64).tolist()

dataC = ((np.array(data2[:]))[:,1:-1]).astype(np.float64).tolist()
dataT = ((np.array(data3[:]))[:,1:-1]).astype(np.float64).tolist()
classDataTrain = ((np.array(data1[:]))[:,-1:]).astype(int).tolist()
classDataClass = ((np.array(data2[:]))[:,-1:]).astype(int).tolist()
classDataTest = ((np.array(data3[:]))[:,-1:]).astype(int).tolist()
dataTraining = []
dataClass = []
'''
dataC = getPCAFeatures(dataC)
dataTrain = getPCAFeatures(dataTrain)
dataT = getPCAFeatures(dataT)

'''
dataC = featureSelection(dataC)
dataTrain = featureSelection(dataTrain)
dataT = featureSelection(dataT)


dataTesting = []
ignoredClass = [] # eliminated Class
#ignoredClass = [8,22,23,25,27,29]

for i in range(len(dataTrain)):
    if (classDataTrain[i][0] not in ignoredClass):
        dataArray = []
        dataArray.append(dataTrain[i])
        dataArray.append(classDataTrain[i][0])
        dataTraining.append(dataArray)

for i in range(len(dataC)):
    if (classDataClass[i][0] not in ignoredClass):
        dataArray2 = []
        dataArray2.append(dataC[i])
        dataArray2.append(classDataClass[i][0])
        dataClass.append(dataArray2)
        #dataTesting.append(dataT[i])
for i in range(len(dataT)):
    dataTesting.append(dataT[i])

wrongClass = 0
for i in range(len(dataTesting)):
    classMin = -1
    minValue = 999999999        
    for j in range(len(dataTraining)):
        jarak = 0
        for k in range(len(dataTraining[j][0])):
            jarak += np.power(dataTesting[i][k] - dataTraining[j][0][k], 2)
        jarak = np.sqrt(jarak)
        '''
        if(i == 0):
            print(jarak)
        '''
        if (jarak < minValue):
            minValue = jarak
            classMin = dataTraining[j][1]
    
    if (classDataTest[i][0] != classMin):
        wrongClass += 1
    '''
    if(i == 0):
        print(minValue, classMin)
    '''
    #print(classDataTest[i][0],classMin)
akurasi = np.divide(float(len(dataTesting) - wrongClass),float(len(dataTesting)))
print(fold, akurasi)
        