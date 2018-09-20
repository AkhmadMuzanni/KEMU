# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 08:31:42 2018

@author: USER
"""

import cv2
import numpy as np
import os

def resizeImg(image):
    #img = cv2.imread(filename)
    small = cv2.resize(image, (0,0),fx=0.1,fy=0.1)
    return small

#strFile = 'D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\Segmentasi\\1.jpg'
Y = 1
for filename in os.listdir("D:\\KULIAH\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\Satu Satu\\"):    
    strFile = 'D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\Dataset Awal\\'+filename
    #strFile = 'D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\Satu Satu\\004_0001_XiaomiRedmiNote4X.jpg'
    rgbImg = cv2.imread(strFile)
    rgbImg = resizeImg(rgbImg)
    
    #rgbImgFloat = rgbImg.astype(np.float64)
    #blueNorm = np.zeros_like(rgbImgFloat[:,:,0])
    #greenNorm = np.zeros_like(rgbImgFloat[:,:,1])
    #redNorm = np.zeros_like(rgbImgFloat[:,:,2])
    #blueNorm = 0
    #greenNorm = 0
    #redNorm = 0
    #cv2.normalize(rgbImgFloat[:,:,0],  blueNorm, 0, 1, cv2.NORM_MINMAX)
    #cv2.normalize(rgbImgFloat[:,:,1],  greenNorm, 0, 1, cv2.NORM_MINMAX)
    #cv2.normalize(rgbImgFloat[:,:,2],  redNorm, 0, 1, cv2.NORM_MINMAX)
    
    lab = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2Lab)
    
    #cv2.imshow('BLUE',blueImage)
    #cv2.imshow('GREEN',greenImage)
    #cv2.imshow('RED',redImage)
    
    first = lab.copy()
    first[:,:,1] = 0
    first[:,:,2] = 0
    second = lab.copy()
    second[:,:,0] = 0
    second[:,:,2] = 0
    third = lab.copy()
    third[:,:,0] = 0
    third[:,:,1] = 0
    
    #cv2.imshow('FIRST',first)
    #cv2.imshow('SECOND',second)
    #cv2.imshow('THIRD',third)
    #print('genji')
    meanb = (np.array(third[:,:,2])).mean() 
    meana = (np.array(second[:,:,1])).mean() 
    meanL = (np.array(first[:,:,0])).mean() 
    if (( meanb < 130) and (meanL > 170) and (meanL < 200)):
        segmen = first[:,:,0]
        #segmen = third[:,:,2]
    else:
        segmen = third[:,:,2]
    #segmen = first[:,:,0]
    #print(len(third))
    #print(len(third[0]))
    #print(len(third[0][0]))
    #if(Y == 3):        
        #cv2.imshow('FIRST',first)
        #cv2.imshow('SECOND',second)
        #cv2.imshow('THIRD',third)
    #segmen = third[:,:,2]
    segmen2 = second[:,:,1]
    x = 0
    for i in range(len(third)):
        for j in range(len(third[i])):
            if(segmen[i][j] < 140):
                segmen[i][j] = 0
            else:
                segmen[i][j] = 255
    for i in range(len(third)):
        for j in range(len(third[i])):
            if(segmen2[i][j] < 140):
                segmen2[i][j] = 0
            else:
                #segmen[i][j] = 255
                segmen2[i][j] = 255
    if (( meanb < 130) and (meanL > 170) and (meanL < 200)):
        segmen = np.bitwise_not(segmen)
    mask = np.bitwise_or(segmen,segmen2)
    if(Y == 3):
        roti = first
        print((np.array(first[:,:,0])).mean())
        print((np.array(second[:,:,1])).mean())
        print((np.array(third[:,:,2])).mean())
    #cv2.imshow('jenis'+str(Y),mask)
    result = rgbImg.copy()
    result[mask == 0] = (255, 255, 255)
    cv2.imshow('jenis'+str(Y),result)
    #cv2.imwrite('D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\Segmentasi\\'+str(Y)+'.jpg',result)
        #cv2.imshow('FIRST',first)
        #cv2.imshow('SECOND',second)
        #cv2.imshow('THIRD',third)
    Y += 1

cv2.waitKey(0)
