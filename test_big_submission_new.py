# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 08:11:42 2018

@author: pablo
"""


import numpy as np
import pandas as pd
import os
os.chdir('/home/pablo/Documents/NucleiCompetition/Nuclei')
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb
from scipy import ndimage
from skimage.morphology import dilation
from keras.models import load_model
from RLE import *

from utils import *



#%%
# Constant variables
TEST_PATH = '/home/pablo/Documents/NucleiCompetition/Nuclei/Data/stage2_test_final'
OUTPUT_PATH = '/home/pablo/Documents/NucleiCompetition/Nuclei/Models/validation_models'
SUBMISSION_PATH = '/home/pablo/Documents/NucleiCompetition/Nuclei/Submissions/'
H = 128
W = 128
C = 3

test_names_all = os.listdir(TEST_PATH)

thresh_lab = 0.5
thresh_cont = 0.5
#%%







for j in range(0,3):
    if j == 0:
        r = range(0,1000)
        s = 0
        e = 1000
    elif j == 1:
        r = range(1000,2000)
        s = 1000
        e = 2000
    else:
        r = range(2000,len(test_names_all))
        s = 2000
        e = len(test_names_all)
    
    test_number = len(r)
    test_names = test_names_all[s:e]

    X_test = [0] * test_number
    pads_h = [0] * test_number
    pads_w = [0] * test_number
    for i in np.arange(0,test_number):
        path = os.path.join(TEST_PATH, test_names[i], 'images', test_names[i] + '.png')
        im = imread(path, as_grey=False)
        if im.ndim < 3:
            im = gray2rgb(im)
        else:
            im = im[:,:,:C]
        X_test[i],pads_h[i],pads_w[i] = pad_test(im)
        print('Loading: '+str(1000*j + i)+' / '+str(len(test_names_all)))


    Y = [0] * test_number
    model = load_model(os.path.join(OUTPUT_PATH,'model_100_2.h5'))
    for i in range(0,test_number):
        print(str(i)+' / '+str(test_number))
        temp_lab, temp_cont = model.predict(np.expand_dims(X_test[i],axis=0),verbose=0)
        temp_lab = temp_lab[:,:,:,0]
        temp_cont = temp_cont[:,:,:,0]
        Y_lab_test = unpad_test(temp_lab[0],pads_h[i],pads_w[i])
        Y_cont_test = unpad_test(temp_cont[0],pads_h[i],pads_w[i])
        Y[i] = (Y_lab_test >= thresh_lab) & (Y_cont_test <= thresh_cont)        
        Y[i] = Y[i].astype(np.int16)  
        Y[i] = ndimage.binary_fill_holes(Y[i])
        Y[i] = label(Y[i])
        Y[i] = dilation(Y[i])
        #X_test[i] = unpad_test(X_test[i],pads_h[i],pads_w[i])    
    
    
        
    

    #Y_test = watershed_segmentation(Y_test)

    prepareSubmission_new(test_names,Y,SUBMISSION_PATH+'gen'+str(j+1)+'.csv')


#%%

results = pd.read_csv(SUBMISSION_PATH+'gen1.csv')
for i in range(1,3):
    results = results.append(pd.read_csv(SUBMISSION_PATH+'gen'+str(i+1)+'.csv'))


ids = results['ImageId']
ids = ids.tolist()
unique_ids = unique(ids)
non_app = findNonAppeared(test_names_all,unique_ids)
sub = pd.DataFrame()
sub['ImageId'] = non_app
sub['EncodedPixels'] = '1 1'
sub.to_csv(SUBMISSION_PATH + 'gen_nonapp.csv', index=False)
results.append(pd.read_csv(SUBMISSION_PATH+'gen_nonapp.csv'))

results.to_csv(SUBMISSION_PATH+'final.csv',index=False)