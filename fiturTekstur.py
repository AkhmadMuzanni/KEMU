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

strFile = 'D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\Satu Satu\\001_0001_XiaomiRedmiNote4X.jpg'
rgbImg = cv2.imread(strFile)
rgbImg = resizeImg(rgbImg)

grayImg = RGBtoGray(rgbImg)

CM = np.zeros((255,255), dtype=int)

cv2.imshow('HASIL',grayImg)
cv2.waitKey(0)