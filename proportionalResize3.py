# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 20:43:43 2018

@author: USER
"""

import os
import cv2
import imutils
import time
import csv
import numpy as np

import fiturWarna as fw
import fiturTekstur as ft
import segmentasiWarna1 as sg


start_time = time.time()

def resizeImg(image):    
    # Set image to landscape position
    if (len(image) > len(image[0])):
        image = imutils.rotate_bound(image, -90)
    # Find ratio of image
    ratio = 500.0/len(image)
    # Resize image based on ratio
    small = cv2.resize(image, (0,0),fx=ratio,fy=ratio)    
    return small

def rename():    
    #print inspect.getfile(inspect.currentframe()) 
    path = "D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\TEST\\"
    tot = os.listdir(path)
    i = 1
    kelas = 1
    for filename in tot:
        if (kelas != int(filename[:3])):
            i = 1
            kelas = int(filename[:3])
        #if filename.endswith(".jpg"):
        #print(str(tot)+str(filename))
        print(str(i).zfill(3)+".jpg")
        os.rename(path+filename, path+filename[:3]+"_"+str(i).zfill(4)+".jpg")
        i += 1

path = "D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\DATASET BALANCE2\\"
pathOutput = "D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\DATA03\\"
tot = os.listdir(path)
X = 1
#kelas = 1
with open('data/realTraining/data03.csv', 'a') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    for filename in tot:
        if (int(filename[:3]) >= 21 and int(filename[:3]) <= 31):
            #if (X == 1):
                
                print(filename)
                
                ## RESIZE ##
                #strFile = 'D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\DATASET BALANCE2\\001_0002.jpg'
                rgbImg = cv2.imread(path+filename)
                #print("baca file")
                #print(time.time()-start_time)
                rgbImg = resizeImg(rgbImg)
                
                ## SEGMENTATION ##
                rgbImg = sg.segmentation(rgbImg)
                
                #print("rotate")
                #print(time.time()-start_time)
                #rotated = imutils.rotate_bound(rgbImg, 90)
                #cv2.imshow('Asli'+str(i),rgbImg)
                #cv2.imwrite(pathOutput+filename,rgbImg)        
                
                ## FEATURE TEXTURE EXTRACTION
                #coMatrix0, coMatrix45, coMatrix90, coMatrix135 = ft.getCoMatrix(rgbImg)
                #cv2.imshow("sasa",rgbImg)
                coMatrix = ft.getCoMatrix(rgbImg)
                #print("coMatrix")
                #print(time.time()-start_time)
                '''    
                sumCoMatrix0 = ft.sumCM(coMatrix0)
                sumCoMatrix45 = ft.sumCM(coMatrix45)
                sumCoMatrix90 = ft.sumCM(coMatrix90)
                sumCoMatrix135 = ft.sumCM(coMatrix135)
                
                meanX0, meanY0, varX0, varY0, energy0, entropy0, contrast0, dissimilarity0, homogeneity0, correlation0 = ft.getFeature(coMatrix0, sumCoMatrix0)
                meanX45, meanY45, varX45, varY45, energy45, entropy45, contrast45, dissimilarity45, homogeneity45, correlation45 = ft.getFeature(coMatrix45, sumCoMatrix45)
                meanX90, meanY90, varX90, varY90, energy90, entropy90, contrast90, dissimilarity90, homogeneity90, correlation90 = ft.getFeature(coMatrix90, sumCoMatrix90)
                meanX135, meanY135, varX135, varY135, energy135, entropy135, contrast135, dissimilarity135, homogeneity135, correlation135 = ft.getFeature(coMatrix135, sumCoMatrix135)
                '''
                meanX0, meanY0, varX0, varY0, energy0, entropy0, contrast0, dissimilarity0, homogeneity0, correlation0 = ft.getFeature(coMatrix[0])
                meanX45, meanY45, varX45, varY45, energy45, entropy45, contrast45, dissimilarity45, homogeneity45, correlation45 = ft.getFeature(coMatrix[1])
                meanX90, meanY90, varX90, varY90, energy90, entropy90, contrast90, dissimilarity90, homogeneity90, correlation90 = ft.getFeature(coMatrix[2])
                meanX135, meanY135, varX135, varY135, energy135, entropy135, contrast135, dissimilarity135, homogeneity135, correlation135 = ft.getFeature(coMatrix[3])
                #print("hitungfitur")
                #print(time.time()-start_time)
                
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
                
                #print("appendfitur")
                
                
                
                #print("segmentasi")
                #print(time.time()-start_time)
                cv2.imwrite(pathOutput+filename,rgbImg)
                #cv2.imshow('Asli'+str(X),segmentImg)
                #print(time.time()-start_time)
                
                ## COLOR FEATURE EXTRACTION ##
                labNorm = fw.convBGRtoLAB(rgbImg)
                lab = np.zeros_like(rgbImg)
                for i in range(len(labNorm)):
                    for j in range(len(labNorm[i])):
                        lab[i][j][0] = labNorm[i][j][0] * 255 / 100
                        lab[i][j][1] = labNorm[i][j][1] + 128
                        lab[i][j][2] = labNorm[i][j][2] + 128
                meanL, varL, skewL, kurtL = fw.getColorMoment(lab[:,:,0])
                meanA, varA, skewA, kurtA = fw.getColorMoment(lab[:,:,1])
                meanB, varB, skewB, kurtB = fw.getColorMoment(lab[:,:,2])
                
                fiturWarna = []
                fiturWarna.append(meanL)
                fiturWarna.append(varL)
                fiturWarna.append(skewL)
                fiturWarna.append(kurtL)
                fiturWarna.append(meanA)
                fiturWarna.append(varA)
                fiturWarna.append(skewA)
                fiturWarna.append(kurtA)
                fiturWarna.append(meanB)
                fiturWarna.append(varB)
                fiturWarna.append(skewB)
                fiturWarna.append(kurtB)
                #print("fitur warna")
                #print(time.time()-start_time)
                
                ## RECORD FEATURE ##
                fitur = []
                fitur.append(filename[:8])
                
                for i in fiturTekstur:
                    fitur.append(i)
                
                for i in fiturWarna:
                    fitur.append(i)
                
                wr.writerow(fitur)
                #print('selesai')                
                print(time.time()-start_time)
            
        X+=1
        #print(X)
cv2.waitKey(0)