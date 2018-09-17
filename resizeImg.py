# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 12:10:59 2018

@author: USER
"""

import cv2

#file('D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\_001. Donat\\001_0001_XiaomiRedmiNote4X.jpg')
img = cv2.imread('D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\Dataset Awal\\016_0007_XiaomiRedmiNote4X.jpg')
#PREPROSESING
def resizeImg(image):
    #img = cv2.imread(filename)
    small = cv2.resize(image, (0,0),fx=0.1,fy=0.1)
    return small

def RGBtoGray(image):
    #img = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def segmentation(image):
    no, segImg = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    return segImg

#filename = 'D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\_001. Donat\\001_0001_XiaomiRedmiNote4X.jpg'

small = resizeImg(img)
gray = RGBtoGray(small)
segImg = segmentation(gray)
cv2.imwrite('test.jpg',gray)
cv2.imshow('Hasil',segImg)
cv2.waitKey(0)