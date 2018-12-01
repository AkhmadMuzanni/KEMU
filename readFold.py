# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 21:05:17 2018

@author: USER
"""

import csv
import os

def read_csv(file_name):
    array_2D = []
    with open(file_name, 'rb') as csvfile:
        read = csv.reader(csvfile, delimiter=';')
        for row in read:
            array_2D.append(row)
    return array_2D

def buildFold(foldTest):
    path = "data/realTraining/fold/"
    #pathOutput = "D:\\KULIAH\\SEMESTER VII\\SKRIPSI - OFFLINE\\DATA01\\"
    tot = os.listdir(path)    
    #kelas = 1
    fold = [] 
    for filename in tot:       
        fold.append(read_csv(path+filename)) # Fold 1
    
    #foldTest = [0]
    dataUji = []
    for i in range(len(foldTest)):
        dataUji.extend(fold[i])
    dataLatih = []
    for i in range(len(fold)):
        if (i in foldTest):
            dataLatih.extend(fold[i])
    return dataLatih, dataUji

foldTest = [0]
buildFold(foldTest)