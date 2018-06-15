# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:49:22 2018

@author: pablo
"""

import numpy as np
import os
os.chdir('/home/pablo/Documents/NucleiCompetition/Nuclei')
from utils import *


TRAIN_PATH = '/home/pablo/Documents/NucleiCompetition/Nuclei/Data/stage1_train'
train_names = os.listdir(TRAIN_PATH)
train_number = len(train_names)

crop = 256

#%%
#Crop the images in blocks of 256x256 pixels

X = np.zeros((1,crop,crop,3)).astype(np.int16)
for im in range(0,train_number):
    print(im)
    path = os.path.join(TRAIN_PATH,train_names[im],'images',train_names[im]+'.png')
    image = imread(path)
    if len(image.shape) < 3:
        image = gray2rgb(image)
    else:
        image = image[:,:,:3]
    h,w = image.shape[:2]
    
    for r in range(0,h-crop,crop):   
        for c in range(0,w-crop,crop):
            X = np.concatenate((X,np.expand_dims(image[r:r+crop,c:c+crop,:],axis=0)),axis=0)
        X = np.concatenate((X,np.expand_dims(image[r:r+crop,-crop:,:],axis=0)),axis=0)
    for c in range(0,w-crop,crop):
        X = np.concatenate((X,np.expand_dims(image[-crop:,c:c+crop,:],axis=0)),axis=0)
    X = np.concatenate((X,np.expand_dims(image[-crop:,-crop:,:],axis=0)),axis=0)
X = X[1:]

np.save('PreprocessedData/X.npy',X)

#%%
# Prepare the labels and contours

for i in range(0,train_number):
    print('Doing train image '+str(i))
    mask_names = os.listdir(os.path.join(TRAIN_PATH,train_names[i],'masks'))
    
    
    mask_final = imread(os.path.join(TRAIN_PATH,train_names[i],'masks',mask_names[0]))
    im_cont = np.zeros(mask_final.shape)
    mask_final = np.zeros((mask_final.shape))
    for mask_name in mask_names:
        mask = imread(os.path.join(TRAIN_PATH,train_names[i],'masks',mask_name))
        mask_final = mask_final + mask
        garbage, contours, garbage = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)        
        if len(contours) > 0:        
            cont = contours[0]
            for j in range(0,cont.shape[0]):
                im_cont[cont[j][0][1],cont[j][0][0]] = 1
    im_cont = im_cont.astype(np.int16)
    mask_final[im_cont==1] = 0
    mask_final = mask_final.astype(np.int16)
    np.save('PreprocessedData/Y_labels/'+train_names[i]+'.npy',mask_final)
    np.save('PreprocessedData/Y_cont/'+train_names[i]+'.npy',im_cont)


#%%
# Crop the labels and contours in blocks of 256x256 pixels

Ylab = np.zeros((1,crop,crop)).astype(np.int8)
Ycont = np.zeros((1,crop,crop)).astype(np.int8)
for im in range(0,train_number):
    print(im)
    #path = os.path.join(TRAIN_PATH,train_names[im],'images',train_names[im]+'.png')
    lab = np.load('PreprocessedData/Y_labels/'+train_names[im]+'.npy').astype(np.int8)
    cont = np.load ('PreprocessedData/Y_cont/'+train_names[im]+'.npy').astype(np.int8)
    h,w = lab.shape[:2]
    for r in range(0,h-crop,crop):   
        for c in range(0,w-crop,crop):
            Ylab = np.concatenate((Ylab,np.expand_dims(lab[r:r+crop,c:c+crop],axis=0)),axis=0)
            Ycont = np.concatenate((Ycont,np.expand_dims(cont[r:r+crop,c:c+crop],axis=0)),axis=0)
        Ylab = np.concatenate((Ylab,np.expand_dims(lab[r:r+crop,-crop:],axis=0)),axis=0)
        Ycont = np.concatenate((Ycont,np.expand_dims(cont[r:r+crop,-crop:],axis=0)),axis=0)
    for c in range(0,w-crop,crop):
        Ylab = np.concatenate((Ylab,np.expand_dims(lab[-crop:,c:c+crop],axis=0)),axis=0)
        Ycont = np.concatenate((Ycont,np.expand_dims(cont[-crop:,c:c+crop],axis=0)),axis=0)
    Ylab = np.concatenate((Ylab,np.expand_dims(lab[-crop:,-crop:],axis=0)),axis=0)
    Ycont = np.concatenate((Ycont,np.expand_dims(cont[-crop:,-crop:],axis=0)),axis=0)

Ylab = Ylab[1:]
Ycont = Ycont[1:]
np.save('PreprocessedData/Ylab.npy',Ylab)
np.save('PreprocessedData/Ycont.npy',Ycont)


#%%
#Check a few images to make sure that everything is ok

X = np.load('PreprocessedData/X.npy')
Ylab = np.load('PreprocessedData/Ylab.npy')
Ycont = np.load('PreprocessedData/Ycont.npy')

#%%

visualize(X,Ycont,965)

