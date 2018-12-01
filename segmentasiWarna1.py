# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 08:31:42 2018

@author: USER
"""

import cv2
import numpy as np
import fiturWarna as fw

def resizeImg(image):
    #img = cv2.imread(filename)
    small = cv2.resize(image, (0,0),fx=0.1,fy=0.1)
    return small

def segmentation(rgbImg):
    blur = cv2.GaussianBlur(rgbImg,(11,11),0)
    
    labNorm = fw.convBGRtoLAB(blur)
    lab = np.zeros_like(rgbImg)
    for i in range(len(labNorm)):
        for j in range(len(labNorm[i])):
            lab[i][j][0] = labNorm[i][j][0] * 255 / 100
            lab[i][j][1] = labNorm[i][j][1] + 128
            lab[i][j][2] = labNorm[i][j][2] + 128
    #blur = cv2.GaussianBlur(lab,(5,5),0)
    
    #lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    
    #l = lab[:,:,0]
    #b = lab[:,:,2]
   # ret3,th3 = cv2.threshold(lab,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
    
    #cv2.imshow('BLUE',blueImage)
    #cv2.imshow('GREEN',greenImage)
    #cv2.imshow('RED',redImage)
    
    first = lab[:,:,0]
    #first[:,:,1] = 0
    #first[:,:,2] = 0
    second = lab[:,:,1]
    #second[:,:,0] = 0
    #second[:,:,2] = 0
    third = lab[:,:,2]
    #third[:,:,0] = 0
    #third[:,:,1] = 0
    
    #cv2.imshow('FIRST',first)
    #cv2.imshow('SECOND',second)
    #cv2.imshow('THIRD',third)
    #print('genji')
    
    #meanb = (np.array(third)).mean()
    #meana = (np.array(second)).mean()
    #meanL = (np.array(first)).mean()            
    stdb = np.std((np.array(third)))
    #stda = np.std((np.array(second)))
    #stdL = np.std((np.array(first)))
    #wr.writerow([meanL,meana,meanb])
    #wr.writerow([stdL,stda,stdb])
    
    #segmen2 = second.copy()
    #print(filename[:8])
    #if (( meanb < 130) and (meanL > 170) and (meanL < 200)):            
    ret = 0
    #ret2 = 0
    th = []
    #th2 = []
    if (stdb<2):
        #print("Dominan Hitam")        
        segmen = first.copy()
        ret,th = cv2.threshold(segmen,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        th = np.bitwise_not(th)
        '''
        for i in range(len(first)):
            for j in range(len(first[i])):
                #if(segmen[i][j] < 140):
                if(segmen[i][j] > 200):
                    segmen[i][j] = 0
                else:
                    segmen[i][j] = 255
        for i in range(len(second)):
            for j in range(len(second[i])):
                #if(segmen2[i][j] < 140):
                if(segmen2[i][j] < 130):
                    segmen2[i][j] = 0
                else:
                    #segmen[i][j] = 255
                    segmen2[i][j] = 255
        #segmen = third[:,:,2]
        '''
    else:
        #print("Dominan Putih")
        segmen = third.copy()
        segmen2 = second.copy()
        ret,th = cv2.threshold(segmen,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #ret2,th2 = cv2.threshold(segmen2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #th = np.bitwise_or(th,th2)
        #cv2.imshow('a'+str(Y),th2)
        '''
        for i in range(len(first)):
            for j in range(len(first[i])):
                #if(segmen[i][j] < 140):
                if(segmen[i][j] < 140):
                    segmen[i][j] = 0
                else:
                    segmen[i][j] = 255
        '''
        for i in range(len(second)):
            for j in range(len(second[i])):
                #if(segmen2[i][j] < 140):
                if(segmen2[i][j] < 135):
                    segmen2[i][j] = 0
                else:
                    #segmen[i][j] = 255
                    segmen2[i][j] = 255
        th = np.bitwise_or(th,segmen2)
                    
    #closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    closing = th
    
    #segmen = first[:,:,0]
    #print(len(third))
    #print(len(third[0]))
    #print(len(third[0][0]))
    #if(Y == 3):        
        #cv2.imshow('FIRST',first)
        #cv2.imshow('SECOND',second)
        #cv2.imshow('THIRD',third)
    #segmen = third[:,:,2]
    
    
    #x = 0
    '''
    for i in range(len(second)):
        for j in range(len(second[i])):
            #if(segmen2[i][j] < 140):
            if(segmen2[i][j] < 130):
                segmen2[i][j] = 0
            else:
                #segmen[i][j] = 255
                segmen2[i][j] = 255
    '''
    #if (stdb<2):
    #if (( meanb < 130) and (meanL > 170) and (meanL < 200)):
        #segmen = np.bitwise_not(segmen)
    #mask = np.bitwise_or(segmen,segmen2)
    #if(Y == 3):
        #roti = first
        #print((np.array(first[:,:,0])).mean())
        #print((np.array(second[:,:,1])).mean())
        #print((np.array(third[:,:,2])).mean())
    #cv2.imshow('jenis'+str(Y),mask)
    
    result = rgbImg[:,:,:]
    result[closing == 0] = (255, 255, 255)
    
    #cv2.imshow('jenis'+str(Y),result)
    #cv2.imshow('jenis'+str(Y),lab)
    #cv2.imshow('otsu'+str(Y),result)
    
    #cv2.imwrite('D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\Ahmad Fauzi A _ Akhmad Muzanni S\\Segmentasi_All_Full\\'+filename[:8]+'.jpg',result)
    #cv2.imshow('FIRST',first)
    #cv2.imshow('SECOND',second)
    #cv2.imshow('THIRD',third)
    #return result
    #print(filename[:8])
    #print(stda)
    #print(ret3)
    #print(cv2.THRESH_OTSU)
    #print(cv2.THRESH_BINARY)
    return result

#cv2.waitKey(0)
