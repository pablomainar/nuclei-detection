"""
Created on Thu Jun  7 12:37:22 2018

This script trains the model using the two-branched version of U-Net: one for segmentation and the other one for contours

@author: pablo
"""

import numpy as np
import os
from utils import * 
from keras_net import create_unet_twoOutputs
from DataGenerator import *
from keras.callbacks import ModelCheckpoint,Callback
from keras.utils.vis_utils import plot_model
import random


# Constant variables
OUTPUT_PATH= './models' #Path where the models will be saved
H = 256 #Height of the training images
W = 256 #Width of the training images
C = 3 #Number of channels of the training images


# Training data should have been preprocessed before running this script. Run first preprocess_data.py.
X = np.load('PreprocessedData/X.npy')
Ylab = np.load('PreprocessedData/Ylab.npy')
Ycont = np.load('PreprocessedData/Ycont.npy')
X = X.astype(np.int16)
Ylab = Ylab.astype(np.int16)
Ycont = Ycont.astype(np.int16)


# Shuffle the data randomly and split it into training and validation set
perm = np.random.permutation(X.shape[0])
X = X[perm]
Ylab = Ylab[perm]
Ycont = Ycont[perm]

X_val = X[:224] 
Ylab_val = Ylab[:224]
Ycont_val = Ycont[:224]
X = X[224:]
Ylab = Ylab[224:]
Ycont = Ycont[224:]


# Set training parameters
batchsize = 8
val_batchsize = 8
nb_epoch = 100
steps_per_epoch = int(X.shape[0]/batchsize)
validation_steps = int(X_val.shape[0]/val_batchsize)


# Callback function to monitorize the losses
class LossHistory(Callback):
    def on_train_begin(self,logs={}):
        self.train_losses = []
        self.val_losses = []
    def on_batch_end(self,batch,logs={}):
        self.train_losses.append(logs.get('loss'))
    def on_epoch_end(self,batch,logs={}):
        self.val_losses.append(logs.get('val_loss'))

        
# Create model
print('Creating model')
model = create_unet_twoOutputs()
plot_model(model,to_file='model_twoOutputs.png',show_shapes=True)
model.compile(optimizer='adam', loss='binary_crossentropy',loss_weights=[1.0,1.0])
model.summary()
model_name = 'model_{epoch:02d}.h5'
checkpointer = ModelCheckpoint(os.path.join(OUTPUT_PATH,model_name), verbose=1, save_best_only=False)
history = LossHistory()

# Train model
results = model.fit_generator(
            generator_twoOutputs(X,Ylab,Ycont,batchsize),
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[checkpointer,history],
            validation_data = generator_twoOutputs_val(X_val,Ylab_val,Ycont_val,val_batchsize),
            validation_steps = validation_steps)

# Save final losses
np.save('train_loss.npy',train_losses)
np.save('val_loss.npy',val_losses)
