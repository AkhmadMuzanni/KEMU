# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:44:49 2018

@author: USER
"""

import cv2
import numpy as np
import os
import fiturWarna as fw

def resizeImg(image):
    #img = cv2.imread(filename)
    small = cv2.resize(image, (0,0),fx=0.1,fy=0.1)
    return small
x = 0
for filename in os.listdir("D:\\KULIAH\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\Satu Satu\\"):    
    if (x < 32):
        strFile = 'D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\Satu Satu\\'+filename
        #strFile = 'D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\Satu Satu\\004_0001_XiaomiRedmiNote4X.jpg'
        rgbImg = cv2.imread(strFile)
        rgbImg = resizeImg(rgbImg)
        labNorm = fw.convBGRtoLAB(rgbImg)
        lab = np.zeros_like(rgbImg)
        for i in range(len(labNorm)):
            for j in range(len(labNorm[i])):
                lab[i][j][0] = np.round(labNorm[i][j][0] * 255 / 100)
                lab[i][j][1] = np.round(labNorm[i][j][1] + 128)
                lab[i][j][2] = np.round(labNorm[i][j][2] + 128)
        meanL, varL, skewL = fw.getColorMoment(labNorm[:,:,0])
        meanA, varA, skewA = fw.getColorMoment(labNorm[:,:,1])
        meanB, varB, skewB = fw.getColorMoment(labNorm[:,:,2])
        print(filename)
        print(meanL)
        print(meanA)
        print(meanB)        
    x+=1
cv2.waitKey(0)

            
