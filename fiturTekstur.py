# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 07:31:50 2018

@author: USER
"""

import cv2
import numpy as np

def RGBtoGray(rgbImg):
    grayImg = np.zeros_like(rgbImg[:,:,0])
    for i in range(len(grayImg)):
        for j in range(len(grayImg[i])):
            grayImg[i][j] = 0.299*rgbImg[i][j][2] + 0.587*rgbImg[i][j][1] + 0.114*rgbImg[i][j][0]
    return grayImg

def resizeImg(image):
    #img = cv2.imread(filename)
    small = cv2.resize(image, (0,0),fx=0.1,fy=0.1)
    return small

def getValue(i, j, grayImg):
    if((i < 0) or (j < 0) or (i >= len(grayImg)) or (j >= len(grayImg[0]))):
        return -1
    else:
        return grayImg[i][j]

def sumCM(coMatrix):
    sumValue = 0
    for i in range(len(coMatrix)):
        for j in range(len(coMatrix[i])):
            sumValue += coMatrix[i][j]
    return sumValue

def getProbMatrix(coMatrix, sumValue):
    probMatrix = coMatrix.copy()
    probMatrix = probMatrix.astype(np.float64)
    #maxValue = np.max(probMatrix)    
    for i in range(len(probMatrix)):
        for j in range(len(probMatrix[i])):
            probMatrix[i][j] = np.divide(probMatrix[i][j], sumValue)
    return probMatrix

# FEATURE 1 : MEAN
def meanX(probMatrix):
    sumTotal = 0
    for i in range(len(probMatrix)):
        sumJ = 0
        for j in range(len(probMatrix[i])):
            sumJ += probMatrix[i][j]
        sumTotal += i*sumJ
    return sumTotal

def meanY(probMatrix):
    sumTotal = 0
    for j in range(len(probMatrix[0])):
        sumI = 0
        for i in range(len(probMatrix)):
            sumI += probMatrix[i][j]
        sumTotal += j*sumI
    return sumTotal

# FEATURE 2 : VARIANS
def variansX(probMatrix, meanX):
    sumTotal = 0
    for i in range(len(probMatrix)):
        sumJ = 0
        for j in range(len(probMatrix[i])):
            sumJ += probMatrix[i][j]
        sumTotal += np.power(i-meanX,2)*sumJ
    return sumTotal

def variansY(probMatrix, meanY):
    sumTotal = 0
    for j in range(len(probMatrix[0])):
        sumI = 0
        for i in range(len(probMatrix)):
            sumI += probMatrix[i][j]
        sumTotal += np.power(j-meanY,2)*sumI
    return sumTotal

# FEATURE 3 : ENERGY
def energy(probMatrix):
    sumTotal = 0
    for i in range(len(probMatrix)):        
        for j in range(len(probMatrix[i])):
            sumTotal += np.power(probMatrix[i][j],2)
    return sumTotal

# FEATURE 4 : ENTROPY
def entropy(probMatrix):
    sumTotal = 0
    for i in range(len(probMatrix)):        
        for j in range(len(probMatrix[i])):
            if (probMatrix[i][j] != 0.0):
                sumTotal += probMatrix[i][j]*np.log(probMatrix[i][j])
    return -sumTotal

# FEATURE 5 : CONTRAST
def contrast(probMatrix):
    sumTotal = 0
    for i in range(len(probMatrix)):        
        for j in range(len(probMatrix[i])):            
            sumTotal += np.power(i-j,2)*probMatrix[i][j]
    return sumTotal

# FEATURE 6 : DISSIMILARITY
def dissimilarity(probMatrix):
    sumTotal = 0
    for i in range(len(probMatrix)):        
        for j in range(len(probMatrix[i])):            
            sumTotal += np.abs(i-j)*probMatrix[i][j]
    return sumTotal

# FEATURE 7 : HOMOGENEITY
def homogeneity(probMatrix):
    sumTotal = 0
    for i in range(len(probMatrix)):        
        for j in range(len(probMatrix[i])):            
            sumTotal += np.divide(probMatrix[i][j],1+np.power(i-j,2))
    return sumTotal

# FEATURE 8 : CORRELATION
def correlation(probMatrix, meanX, meanY, varX, varY):
    sumTotal = 0
    for i in range(len(probMatrix)):        
        for j in range(len(probMatrix[i])):            
            sumTotal += np.divide((i-meanY)*(j-meanX)*probMatrix[i][j],varX*varY)
    return sumTotal


strFile = 'D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\Satu Satu\\001_0001_XiaomiRedmiNote4X.jpg'
rgbImg = cv2.imread(strFile)
rgbImg = resizeImg(rgbImg)

grayImg = RGBtoGray(rgbImg)

coMatrix0 = np.zeros((255,255), dtype=int)
# degree 0
for i in range(len(grayImg)):
    for j in range(len(grayImg[i])):
        if (getValue(i, j+1, grayImg) != -1):
            coMatrix0[getValue(i,j,grayImg)][getValue(i,j+1,grayImg)] += 1

coMatrix45 = np.zeros((255,255), dtype=int)
# degree 45
for i in range(len(grayImg)):
    for j in range(len(grayImg[i])):
        if (getValue(i-1, j+1, grayImg) != -1):
            coMatrix45[getValue(i,j,grayImg)][getValue(i-1,j+1,grayImg)] += 1

coMatrix90 = np.zeros((255,255), dtype=int)
# degree 90
for i in range(len(grayImg)):
    for j in range(len(grayImg[i])):
        if (getValue(i-1, j, grayImg) != -1):
            coMatrix90[getValue(i,j,grayImg)][getValue(i-1,j,grayImg)] += 1

coMatrix135 = np.zeros((255,255), dtype=int)
# degree 135
for i in range(len(grayImg)):
    for j in range(len(grayImg[i])):
        if (getValue(i-1, j-1, grayImg) != -1):
            coMatrix135[getValue(i,j,grayImg)][getValue(i-1,j-1,grayImg)] += 1

sumCoMatrix = sumCM(coMatrix0)

probMatrix0 = getProbMatrix(coMatrix0, sumCoMatrix)
meanX0 = meanX(probMatrix0)
meanY0 = meanY(probMatrix0)
varX0 = variansX(probMatrix0, meanX0)
varY0 = variansY(probMatrix0, meanY0)
energy0 = energy(probMatrix0)
entropy0 = energy(probMatrix0)
contrast0 = contrast(probMatrix0)
dissimilarity0 = dissimilarity(probMatrix0)
homogeneity0 = homogeneity(probMatrix0)
correlation0 = correlation(probMatrix0, meanX0, meanY0, varX0, varY0)

probMatrix45 = getProbMatrix(coMatrix45, sumCoMatrix)
meanX45 = meanX(probMatrix45)
meanY45 = meanY(probMatrix45)
varX45 = variansX(probMatrix45, meanX45)
varY45 = variansY(probMatrix45, meanY45)
energy45 = energy(probMatrix45)
entropy45 = energy(probMatrix45)
contrast45 = contrast(probMatrix45)
dissimilarity45 = dissimilarity(probMatrix45)
homogeneity45 = homogeneity(probMatrix45)
correlation45 = correlation(probMatrix45, meanX45, meanY45, varX45, varY45)

probMatrix90 = getProbMatrix(coMatrix90, sumCoMatrix)
meanX90 = meanX(probMatrix90)
meanY90 = meanY(probMatrix90)
varX90 = variansX(probMatrix90, meanX90)
varY90 = variansY(probMatrix90, meanY90)
energy90 = energy(probMatrix90)
entropy90 = energy(probMatrix90)
contrast90 = contrast(probMatrix90)
dissimilarity90 = dissimilarity(probMatrix90)
homogeneity90 = homogeneity(probMatrix90)
correlation90 = correlation(probMatrix90, meanX90, meanY90, varX90, varY90)

probMatrix135 = getProbMatrix(coMatrix135, sumCoMatrix)
meanX135 = meanX(probMatrix135)
meanY135 = meanY(probMatrix135)
varX135 = variansX(probMatrix135, meanX135)
varY135 = variansY(probMatrix135, meanY135)
energy135 = energy(probMatrix135)
entropy135 = energy(probMatrix135)
contrast135 = contrast(probMatrix135)
dissimilarity135 = dissimilarity(probMatrix135)
homogeneity135 = homogeneity(probMatrix135)
correlation135 = correlation(probMatrix135, meanX135, meanY135, varX135, varY135)

fiturTekstur = []
fiturTekstur.append(meanX0)
fiturTekstur.append(meanY0)
fiturTekstur.append(varX0)
fiturTekstur.append(varY0)
fiturTekstur.append(energy0)
fiturTekstur.append(entropy0)
fiturTekstur.append(contrast0)
fiturTekstur.append(dissimilarity0)
fiturTekstur.append(homogeneity0)
fiturTekstur.append(correlation0)

fiturTekstur.append(meanX45)
fiturTekstur.append(meanY45)
fiturTekstur.append(varX45)
fiturTekstur.append(varY45)
fiturTekstur.append(energy45)
fiturTekstur.append(entropy45)
fiturTekstur.append(contrast45)
fiturTekstur.append(dissimilarity45)
fiturTekstur.append(homogeneity45)
fiturTekstur.append(correlation45)

fiturTekstur.append(meanX90)
fiturTekstur.append(meanY90)
fiturTekstur.append(varX90)
fiturTekstur.append(varY90)
fiturTekstur.append(energy90)
fiturTekstur.append(entropy90)
fiturTekstur.append(contrast90)
fiturTekstur.append(dissimilarity90)
fiturTekstur.append(homogeneity90)
fiturTekstur.append(correlation90)

fiturTekstur.append(meanX135)
fiturTekstur.append(meanY135)
fiturTekstur.append(varX135)
fiturTekstur.append(varY135)
fiturTekstur.append(energy135)
fiturTekstur.append(entropy135)
fiturTekstur.append(contrast135)
fiturTekstur.append(dissimilarity135)
fiturTekstur.append(homogeneity135)
fiturTekstur.append(correlation135)

print(fiturTekstur)


#print(CM)

#print(grayImg[0][1])


       

cv2.imshow('HASIL',grayImg)
cv2.waitKey(0)