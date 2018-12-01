# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 22:20:49 2018

@author: USER
"""

import cv2
import imutils

def resizeImg(image):    
    # Set image to landscape position
    if (len(image) > len(image[0])):
        image = imutils.rotate_bound(image, -90)
    # Find ratio of image
    ratio = 500.0/len(image)
    # Resize image based on ratio
    small = cv2.resize(image, (0,0),fx=ratio,fy=ratio)    
    return small
    

strFile = 'D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\DATASET BALANCE2\\001_0002.jpg'
rgbImg = cv2.imread(strFile)
rgbImg = resizeImg(rgbImg)
#rotated = imutils.rotate_bound(rgbImg, 90)
cv2.imshow('Asli',rgbImg)
print(len(rgbImg))
print(len(rgbImg[0]))
#cv2.imshow('Rotated',rotated)
cv2.waitKey(0)