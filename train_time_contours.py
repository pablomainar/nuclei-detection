# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:37:22 2018

@author: pablo
"""

import numpy as np
import pandas as pd
import os
os.chdir('/home/pablo/Documents/NucleiCompetition/Nuclei')

from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb
from skimage.morphology import label

import matplotlib.pyplot as plt

import cv2

from utils import * 
from keras_net import create_unet,create_unet_twoOutputs
from DataGenerator import *

from keras.models import Model, load_model, Sequential
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint,Callback
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras import Sequential

import tensorflow
import random


# Constant variables
TRAIN_PATH = '/home/pablo/Documents/NucleiCompetition/Nuclei/Data/stage1_train'
OUTPUT_PATH = '/home/pablo/Documents/NucleiCompetition/Nuclei/Models/validation_models'
H = 256
W = 256
C = 3



X = np.load('PreprocessedData/X.npy')
Ylab = np.load('PreprocessedData/Ylab.npy')
Ycont = np.load('PreprocessedData/Ycont.npy')

X = X.astype(np.int16)
Ylab = Ylab.astype(np.int16)
Ycont = Ycont.astype(np.int16)

#%%

#perm = np.random.permutation(X.shape[0])
perm = np.load('validation_perm.npy')
X = X[perm]
Ylab = Ylab[perm]
Ycont = Ycont[perm]

#%%

X_val = X[:224]
Ylab_val = Ylab[:224]
Ycont_val = Ycont[:224]
X = X[224:]
Ylab = Ylab[224:]
Ycont = Ycont[224:]


#%%


batchsize = 8
val_batchsize = 224
nb_epoch = 100
#steps_per_epoch = 3 * X_train.shape[0]/batchsize
steps_per_epoch = 150

#model = create_unet_twoOutputs()
model = load_model(os.path.join(OUTPUT_PATH,'model_75.h5'))
plot_model(model,to_file='model_twoOutputs.png',show_shapes=True)
model.compile(optimizer='adam', loss='binary_crossentropy',loss_weights=[1.0,1.0])
model.summary()

#%%

class LossHistory(Callback):
    def on_train_begin(self,logs={}):
        self.train_losses = []
        self.val_losses = []
    def on_batch_end(self,batch,logs={}):
        self.train_losses.append(logs.get('loss'))
    def on_epoch_end(self,batch,logs={}):
        self.val_losses.append(logs.get('val_loss'))

model_name = 'model_{epoch:02d}_2.h5'
checkpointer = ModelCheckpoint(os.path.join(OUTPUT_PATH,model_name), verbose=1, save_best_only=False)
history = LossHistory()
#results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=10,callbacks=[checkpointer])
results = model.fit_generator(
            generator_twoOutputs(X,Ylab,Ycont,batchsize),
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[checkpointer,history],
            validation_data = generator_twoOutputs_val(X_val,Ylab_val,Ycont_val,val_batchsize),
            validation_steps = 112)

np.save('train_loss.npy',train_losses)
np.save('val_loss.npy',val_losses)
