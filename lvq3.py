# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 22:05:22 2018

@author: USER
"""
import numpy as np
import csv
import time
from datetime import datetime
import os

#dataClass = [[[1.0,1.0,0.0,0.0],0],[[0.0,0.0,0.0,1.0],1]]
#dataTraining = [[[0.0,0.0,1.0,1.0],1],[[1.0,0.0,0.0,0.0],0],[[0.0,1.0,1.0,0.0],1]]

alpha_init = 0.1
alpha = alpha_init
pengali_alpha = 0.8
m = 0.1
beta = m*alpha
#epsilon= 0.35
epsilon2= 0.4
start_time = time.time()
iterasi_maksimal = 2
alpha_minimal = 0.000001
iterasi = 1
#epsilon2= 0.8

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
    eliminated = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40] # Warna
    #eliminated = [41,42,43,44,45,46,47,48,49,50,51,52] # Haralick
    
    #eliminated = [4,5,14,15,24,25,26,34,35] # Relief 
    #eliminated = [1,2,9,10,11,12,19,20,21,22,29,30,31,32,39,40,41,42,44,45,46,48,49] # Korelasi
    #eliminated = [1,2,3,5,6,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,43,46,47,48,49,50,51,52] # CFS
    #eliminated = [] # Tanpa Seleksi
    for i in range(len(res)):
        if ((i+1) not in eliminated):
            resJ = []
            for j in range(len(res[i])):
                resJ.append(res[i][j])
            result.append(resJ)
    return np.transpose(result).tolist()

def getPCAFeatures(listFeatures):    
    res = listFeatures
    result = []    
    for i in range(len(res)):
        fiturPCA = []        
        fiturPCA.append((0.209*res[i][13])+(0.209*res[i][12])+(0.208*res[i][32])+(0.208*res[i][33])+(0.208*res[i][17]))
        fiturPCA.append((-0.293*res[i][11])-(0.293*res[i][10])-(0.293*res[i][30])-(0.293*res[i][31])-(0.292*res[i][21]))
        fiturPCA.append((-0.301*res[i][5])-(0.301*res[i][4])-(0.299*res[i][25])-(0.299*res[i][24])-(0.297*res[i][34]))
        fiturPCA.append((-0.386*res[i][47])-(0.385*res[i][45])-(0.352*res[i][49])-(0.352*res[i][43])-(0.345*res[i][51]))
        fiturPCA.append((0.348*res[i][18])+(0.345*res[i][28])+(0.344*res[i][8])+(0.341*res[i][38])-(0.264*res[i][40]))
        fiturPCA.append((0.391*res[i][39])+(0.37*res[i][19])+(0.369*res[i][9])+(0.366*res[i][29])-(0.283*res[i][8]))
        fiturPCA.append((-0.41*res[i][41])-(0.393*res[i][46])+(0.373*res[i][51])-(0.349*res[i][43])-(0.34*res[i][40]))
        fiturPCA.append((0.607*res[i][40])-(0.53*res[i][42])+(0.22*res[i][45])+(0.212*res[i][47])-(0.208*res[i][43]))
        result.append(fiturPCA)
    return result

#data1 = read_csv('data/realTraining/dataTrain80204.csv') # Data Training
#data3 = read_csv('data/realTraining/dataTest80204.csv') # Data Testing
data2 = read_csv('data/realTraining/dataClass80204.csv') # Data Class (Vector Reference)
testFold = [9]
fold = 'fold9'

result = []
result.append('fold='+str(fold))
result.append('alpha init = '+str(alpha_init))
result.append('pengali_alpha = '+str(pengali_alpha))
result.append('m = '+str(m))
result.append('epsilon = '+str(epsilon2))
result.append('iterasi_maksimal = '+str(iterasi_maksimal))
result.append('alpha_minimal = '+str(alpha_minimal))
#result.append('fitur = all')
#result.append('fitur = tekstur')
result.append('fitur = warna')
#result.append('waktu = '+str(waktu))
for i in result:
    print(i)

data1, data3 = buildFold(testFold)
'''
data1 = read_csv('data/realTraining/dataTrain9010.csv') # Data Training
data3 = read_csv('data/realTraining/dataTest9010.csv') # Data Testing
data2 = read_csv('data/realTraining/dataClass80204.csv') # Data Class (Vector Reference)
'''

'''
data1 = read_csv('data/pengujian124/dataTrain124.csv') # Data Training
data2 = read_csv('data/pengujian124/dataClass124.csv') # Data Class (Vector Reference)
data3 = read_csv('data/pengujian124/dataTest124.csv') # Data Testing
'''
'''
data1 = read_csv('data/dataTrainAll50.csv') # Data Training
data2 = read_csv('data/dataClassAllTL.csv') # Data Class (Vector Reference)
data3 = read_csv('data/dataTestAll50.csv') # Data Testing

data1 = read_csv('datatraining.csv') # Data Training
data2 = read_csv('refvector.csv') # Data Class (Vector Reference)
data3 = read_csv('datatesting.csv') # Data Testing
'''
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

#for x in range(iterasi):
while(iterasi <= iterasi_maksimal and alpha >= alpha_minimal):
     
    # Find Euclidean Distance
    jarak = np.zeros(len(dataClass),dtype=np.float64)
    for i in range(len(dataTraining)):
        for j in range(len(dataClass)):
            jarak[j] = 0
            for k in range(len(dataTraining[i][0])):
                jarak[j] += np.power(dataTraining[i][0][k] - weightMatrix[j][k],2)
            jarak[j] = np.sqrt(jarak[j])
    
        jVal = []
        for j in range(len(dataClass)):
            jVal.append([jarak[j],j])        
        jVal.sort(key=takeFirst)
        
        t = dataTraining[i][1]
        yc1 = int(jVal[0][1]) # The nearest class from the data training 
        yc2 = int(jVal[1][1]) # The second nearest class from the data training 
        dc1 = jVal[0][0] # The distance of the nearest class from the data training 
        dc2 = jVal[1][0] # The distance of the second nearest class from the data training 
    
    
        for j in range(len(weightMatrix)):
            for k in range(len(weightMatrix[j])):                
                
                
                if ( (min(np.divide(dc1,dc2),np.divide(dc2,dc1)) > ((1-epsilon2)*(1+epsilon2)))):
                    if ((classDataClass[yc1][0] == t and classDataClass[yc2][0] == t) ):                        
                        #print(str(t)+" "+str(yc1)+" "+str(yc2))
                        countLVQ3 += 1
                        if(j == yc1 or j == yc2):
                            weightMatrix[j][k] = (1-beta)*weightMatrix[j][k] + beta*dataTraining[i][0][k]
                    elif((classDataClass[yc1][0] == t and classDataClass[yc2][0] != t) or (classDataClass[yc1][0] != t and classDataClass[yc2][0] == t)):
                        countLVQ21 += 1
                        if(j == yc1):
                            if(t == classDataClass[yc1][0]):
                                weightMatrix[j][k] = (1-alpha)*weightMatrix[j][k] + alpha*dataTraining[i][0][k]
                            else:
                                weightMatrix[j][k] = (1+alpha)*weightMatrix[j][k] - alpha*dataTraining[i][0][k]
                        elif(j == yc2):
                            if(t == classDataClass[yc2][0]):
                                weightMatrix[j][k] = (1-alpha)*weightMatrix[j][k] + alpha*dataTraining[i][0][k]
                            else:
                                weightMatrix[j][k] = (1+alpha)*weightMatrix[j][k] - alpha*dataTraining[i][0][k]               
                '''
                elif (j == yc1):
                    countLVQ1 += 1
                    if(classDataClass[j][0] == t):
                        weightMatrix[j][k] = (1-alpha)*weightMatrix[j][k] + alpha*dataTraining[i][0][k]
                    else:
                        weightMatrix[j][k] = (1+alpha)*weightMatrix[j][k] - alpha*dataTraining[i][0][k]
                '''    
                        
    alpha = pengali_alpha*alpha
    iterasi += 1

            
# Testing
#dataTest = [0.0,0.0,1.0,1.0]
wrongClass = 0
#with open('data/realTesting/testingResultMValueAll04.csv', 'a') as myfile:
with open('data/crossValidation/akurasi/'+fold+'Warna.csv', 'a') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    for z in range(len(dataTesting)):
        testing = classDataTest[z][0]-1
        dataTest = dataTesting[testing]
        
        classResult = -1
        minValue = 999999999    
        for i in range(len(weightMatrix)):
            sumValue = 0
            for j in range(len(weightMatrix[i])):
                sumValue += np.power(dataTesting[z][j] - weightMatrix[i][j],2)
            if (sumValue < minValue):
                minValue = sumValue
                #classResult = i
                classResult = classDataClass[i][0]-1
        result = [testing,classResult]
        wr.writerow(result)
        if (testing != classResult):
            #print('Real Class   = '+str(testing)+' -> Class Result = '+str(classResult)+' -> Min Value = '+str(minValue))
            wrongClass+=1
akurasi = np.divide(float(len(dataTesting) - wrongClass),float(len(dataTesting)))
waktu = time.time()-start_time


with open('data/crossValidation/akurasi/RESULT.csv', 'a') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    result = [fold,akurasi]
    wr.writerow(result)

#print(countLVQ1)
#print(countLVQ2)
#print(countLVQ21)
#print(countLVQ3)
with open('data/crossValidation/akurasi/TIME_COMPUTATION.csv', 'a') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    date = datetime.now()
    result.append(date)
    result.append('alpha akhir = '+str(alpha))
    result.append('iterasi = '+str(iterasi))
    print('alpha akhir = '+str(alpha))
    print('iterasi = '+str(iterasi))
    
    wr.writerow(result)
   

