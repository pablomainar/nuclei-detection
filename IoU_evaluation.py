# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:27:04 2018

@author: pablo
"""


import numpy as np
import pandas as pd
import os
os.chdir('/home/pablo/Documents/NucleiCompetition/Nuclei')
from scipy.stats import mode
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb
from skimage.morphology import label,opening,closing,erosion,dilation
from scipy import ndimage

from RLE import *
from utils import *

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import cv2

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow

import random

# Constant variables
TEST_PATH = '/home/pablo/Documents/NucleiCompetition/Nuclei/Data/stage1_train'
OUTPUT_PATH = '/home/pablo/Documents/NucleiCompetition/Nuclei/Models/validation_models'
C = 3

# Load the testing data
test_names = os.listdir(TEST_PATH)
random.shuffle(test_names)
test_number = len(test_names)
test_number = 2



X_test = [0] * test_number
Y_true = [0] * test_number
pads_h = [0] *test_number
pads_w = [0] *test_number
for i in np.arange(0,test_number):
    print('Loading test image ' + str(i))
    path = os.path.join(TEST_PATH, test_names[i], 'images', test_names[i] + '.png')
    im = imread(path, as_grey=False)
    if im.ndim < 3:
        im = gray2rgb(im)
    else:
        im = im[:,:,:C]
    X_test[i],pads_h[i],pads_w[i] = pad_test(im)
    
    path = os.path.join(TEST_PATH,test_names[i],'masks','*.png')
    masks = skimage.io.imread_collection(path).concatenate()
    Y_true[i] = masks
#%%


Y_lab_test = [0] * test_number
Y_cont_test = [0] * test_number
model = load_model(os.path.join(OUTPUT_PATH,'model_45_2.h5'))
for i in range(0,test_number):
    print('Predicting test image '+str(i))
    temp_lab, temp_cont = model.predict(np.expand_dims(X_test[i],axis=0),verbose=0)
    temp_lab = temp_lab[:,:,:,0]
    temp_cont = temp_cont[:,:,:,0]
    Y_lab_test[i] = unpad_test(temp_lab[0],pads_h[i],pads_w[i])
    Y_cont_test[i] = unpad_test(temp_cont[0],pads_h[i],pads_w[i])
    X_test[i] = unpad_test(X_test[i],pads_h[i],pads_w[i])
    
#%%
    
    
    
thresh_lab = 0.5
#avg = []
thresh_cont_range = np.arange(0.3,1.01,0.1)
thresh_lab_range = np.arange(0.3,1.0,0.1)
#thresh_lab_range = [0.5]
#thresh_cont_range = [0.5]

r = -1
avg = np.zeros((len(thresh_lab_range),len(thresh_cont_range)))
for thresh_lab in thresh_lab_range:
    c = -1
    r = r + 1
    for thresh_cont in thresh_cont_range:
        c = c + 1
        print('Evaluating with thresh_cont '+str(thresh_cont)+' and thresh_lab '+str(thresh_lab))
        Y_pred = [0] * test_number
        for i in range(0,test_number):
            lab = Y_lab_test[i]
            cont = Y_cont_test[i]
            Y_pred[i] = (lab >= thresh_lab) & (cont <= thresh_cont)
            Y_pred[i] = Y_pred[i].astype(np.int16)      
            Y_pred[i] = ndimage.binary_fill_holes(Y_pred[i])
            #Y_pred[i] = opening(Y_pred[i]) 
            #Y_pred[i] = closing(Y_pred[i]) 
            #Y_pred[i] = erosion(Y_pred[i])
            #Y_pred[i] = watershed_segmentation(np.expand_dims(Y_pred[i],0))[0]
            Y_pred[i] = label(Y_pred[i])
            Y_pred[i] = dilation(Y_pred[i])
        #avg.append(computeIoU(X_test,Y_true,Y_pred,verbose=0,return_all_pred=False))
        avg[r,c] = computeIoU(X_test,Y_true,Y_pred,verbose=0,return_all_pred=False)
#%%
             
        
        
#%%
plt.plot(thresh_cont_range,avg)

#%%
fig = plt.figure()
ax = fig.gca(projection='3d')
X,Y = np.meshgrid(thresh_cont_range,thresh_lab_range)
surf = ax.plot_surface(thresh_cont_range,thresh_lab_range,avg,cmap=cm.coolwarm)
plt.show()
#%%

avg = computeIoU(X_test,Y_true,Y_pred,verbose=1)
    
#%%
n = 5
visualize(X_test[n],Y_pred[n])
computeIoU(np.expand_dims(X_test[n],axis=0),np.expand_dims(Y_true[n],axis=0),np.expand_dims(Y_pred[n],axis=0),verbose=1)

#%%

h_true = []
h_pred =[]
for i in range(0,test_number):
    h_true.append(Y_true[i].shape)
    h_pred.append(Y_pred[i].shape)
    

#%%
X = X_test
Ytrue = Y_true
Ypred = Y_pred
num_samples = len(X)
all_prec = []
for i in range(0,num_samples):
    image = X[i]
    masks = Ytrue[i]
    y_pred = Ypred[i]
    #masks_pred = Y_pred[i]

    height, width, _ = image.shape
    num_masks = masks.shape[0]
    
    # Make a ground truth label image (pixel value is index of object label)
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = index + 1
    
    
    
    
    
    
    
    
    
    
    

