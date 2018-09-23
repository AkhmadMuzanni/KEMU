# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 20:19:50 2018

@author: USER
"""

import cv2
import fiturTekstur as ft
import fiturWarna as fw
import csv
import segmentasiWarna as sg
import numpy as np
import time
import os

def resizeImg(image):
    #img = cv2.imread(filename)
    small = cv2.resize(image, (0,0),fx=0.1,fy=0.1)
    return small
start_time = time.time()
np.seterr(all='warn')
Y = 1
for filename in os.listdir("D:\\KULIAH\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\All\\"):
    if (Y < 3):
    
    #filename = '001_0001_XiaomiRedmiNote4X.jpg'
    #strFile = 'D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\All\\001_0001_XiaomiRedmiNote4X.jpg'
        strFile = 'D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\All\\'+filename
        rgbImg = cv2.imread(strFile)
        rgbImg = resizeImg(rgbImg)
        cv2.imshow('ASLI',rgbImg)
        print(filename[:8])
        
        
        # TEXTURE FEATURE EXTRACTION
        coMatrix0, coMatrix45, coMatrix90, coMatrix135 = ft.getCoMatrix(rgbImg)
        sumCoMatrix = ft.sumCM(coMatrix0)
        meanX0, meanY0, varX0, varY0, energy0, entropy0, contrast0, dissimilarity0, homogeneity0, correlation0 = ft.getFeature(coMatrix0, sumCoMatrix)
        meanX45, meanY45, varX45, varY45, energy45, entropy45, contrast45, dissimilarity45, homogeneity45, correlation45 = ft.getFeature(coMatrix45, sumCoMatrix)
        meanX90, meanY90, varX90, varY90, energy90, entropy90, contrast90, dissimilarity90, homogeneity90, correlation90 = ft.getFeature(coMatrix90, sumCoMatrix)
        meanX135, meanY135, varX135, varY135, energy135, entropy135, contrast135, dissimilarity135, homogeneity135, correlation135 = ft.getFeature(coMatrix135, sumCoMatrix)
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
        #print(fiturTekstur)
        
        # COLOR FEATURE EXTRACTION
        #Lab = fw.convBGRtoLAB(rgbImg)
        
        
        
        segmentImg = sg.segmentation(rgbImg)
        
        meanL, varL, skewL = fw.getColorMoment(segmentImg[:,:,0])
        meanA, varA, skewA = fw.getColorMoment(segmentImg[:,:,1])
        meanB, varB, skewB = fw.getColorMoment(segmentImg[:,:,2])
        
        #meanL, varL, skewL, kurtL = fw.getColorMoment(segmentImg[:,:,0])
        #meanA, varA, skewA, kurtA = fw.getColorMoment(segmentImg[:,:,1])
        #meanB, varB, skewB, kurtB = fw.getColorMoment(segmentImg[:,:,2])
        
        fiturWarna = [filename[:8]]
        fiturWarna.append(meanL)
        fiturWarna.append(varL)
        fiturWarna.append(skewL)
        #fiturWarna.append(kurtL)
        fiturWarna.append(meanA)
        fiturWarna.append(varA)
        fiturWarna.append(skewA)
        #fiturWarna.append(kurtA)
        fiturWarna.append(meanB)
        fiturWarna.append(varB)
        fiturWarna.append(skewB)
        #fiturWarna.append(kurtB)
        #print(fiturWarna)
        
        
        cv2.imshow('SEGMENTASI',segmentImg)
        
        fitur = []
        fitur = fitur.extend(fiturTekstur)
        fitur = fitur.extend(fiturWarna)
        with open('data/dataTraining.csv', 'a') as myfile:
            wr = csv.writer(myfile, delimiter=',')            
            wr.writerow(fitur)
    Y+=1

print(time.time() - start_time)
cv2.waitKey(0)