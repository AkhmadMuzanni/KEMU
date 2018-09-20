# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 12:10:59 2018

@author: USER
"""

import cv2
import os

#file('D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\_001. Donat\\001_0001_XiaomiRedmiNote4X.jpg')

#PREPROSESING
def resizeImg(image):
    #img = cv2.imread(filename)
    small = cv2.resize(image, (0,0),fx=0.1,fy=0.1)
    return small

def RGBtoGray(image):
    #img = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def gaussianSegmentation(image):
    #no, segImg = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    segImg = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return segImg

def otsuSegmentation(image):
    no, segImg = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    #segImg = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return segImg

#filename = 'D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\_001. Donat\\001_0001_XiaomiRedmiNote4X.jpg'
i = 0
for filename in os.listdir("D:\\KULIAH\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\Satu Satu\\"):
    if(i < 31):
        #img = cv2.imread('D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\Dataset Awal\\002_0001_XiaomiRedmiNote4X.jpg')
        strFile = 'D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\Dataset Awal\\'+filename
        img = cv2.imread(strFile)
        
        small = resizeImg(img)
        smalla = cv2.bilateralFilter(small,9,75,75)
        #smallb = cv2.bilateralFilter(smalla,9,75,75)
        gray = RGBtoGray(smalla)
        segImg = gaussianSegmentation(gray)
        
        #seg2 = otsuSegmentation(segImg)
        median = cv2.medianBlur(segImg,5)
        neg = cv2.bitwise_not(median)
        #closing = cv2.morphologyEx(median, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
        erosion = cv2.erode(median,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations = 30)
        erosion = cv2.dilate(erosion,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations = 30)
        #erosion = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        erosion = cv2.erode(erosion,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations = 29)
        erosion = cv2.dilate(erosion,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations = 30)
        op = cv2.bitwise_and(erosion,neg)
        
        #erosion = cv2.erode(median,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations = 20)
        #erosion = cv2.dilate(erosion,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations = 20)
        
        
        #closing = cv2.morphologyEx(median, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
        #closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
        #closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
        #closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
        #closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
        
        
        cv2.imwrite('test.jpg',gray)
        blueImage = small.copy()
        blueImage[:,:,1] = 0
        blueImage[:,:,2] = 0
        greenImage = small.copy()
        greenImage[:,:,0] = 0
        greenImage[:,:,2] = 0
        redImage = small.copy()
        redImage[:,:,0] = 0
        redImage[:,:,1] = 0
        
        result = small.copy()
        result[erosion == 255] = (255, 255, 255)
        
        i += 1
        #cv2.imshow('Asli'+str(i),small)
        
        #cv2.imshow('Blur',smalla)
        #cv2.imshow('Gray',gray)
        #cv2.imshow('Gaussian',erosion)
        #cv2.imshow('Gaussian2',median)
        #cv2.imshow('Median'+str(i),median)
        #cv2.imshow('Closing'+str(i),erosion)
        #cv2.imshow('Negatif'+str(i),neg)
        #cv2.imshow('Operasi'+str(i),op)
        cv2.imshow('hasil'+str(i),result)
        #cv2.imwrite('D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\Segmentasi\\'+str(i)+'.jpg',result)
        #cv2.imshow('RED',redImage)
        #cv2.imshow('GREEN',greenImage)
        #cv2.imshow('BLUE',blueImage)
cv2.waitKey(0)