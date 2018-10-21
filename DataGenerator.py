"""
Created on Mon Jun  4 20:06:10 2018

This script constructs the generator used in training

@author: pablo
"""
import numpy as np
from utils import augmentData,augmentDataTwoOutputs

def generator(X,Y,batch_size):
    samples = X.shape[0]
    H = X.shape[1]
    W = X.shape[2]
    C = X.shape[3]
    feat = np.zeros((batch_size,H,W,C))
    labels = np.zeros((batch_size,H,W))
    while True:
        random_im = np.random.choice(range(0,samples),batch_size)
        feat,labels = augmentData(X[random_im],Y[random_im])      
        yield feat.astype(np.int16),np.expand_dims(labels,axis=-1).astype(np.int16)

def generator_twoOutputs(X,Ylab,Ycont,batch_size):
    samples = X.shape[0]
    H = X.shape[1]
    W = X.shape[2]
    C = X.shape[3]
    feat = np.zeros((batch_size,H,W,C))
    labels = np.zeros((batch_size,H,W))
    contours = np.zeros((batch_size,H,W))
    while True:
        random_im = np.random.choice(range(0,samples),batch_size)
        feat,labels,contours = augmentDataTwoOutputs(X[random_im],Ylab[random_im],Ycont[random_im])      
        yield feat.astype(np.int16),{'labels':np.expand_dims(labels,axis=-1).astype(np.int16),'contours':np.expand_dims(contours,axis=-1).astype(np.int16)}


def generator_twoOutputs_val(X,Ylab,Ycont,batch_size):
    samples = X.shape[0]
    H = X.shape[1]
    W = X.shape[2]
    C = X.shape[3]
    feat = np.zeros((2,H,W,C))
    labels = np.zeros((2,H,W))
    contours = np.zeros((2,H,W))
    while True:
        for i in range(0,samples,2):           
            feat = X[i:i+2]  
            labels = Ylab[i:i+2]
            contours = Ycont[i:i+2]
            yield feat.astype(np.int16),{'labels':np.expand_dims(labels,axis=-1).astype(np.int16),'contours':np.expand_dims(contours,axis=-1).astype(np.int16)}


def generator_list(X,Y,batch_size):
    samples = X.shape[0]
    H = X.shape[1]
    W = X.shape[2]
    C = X.shape[3]
    feat = [np.zeros((H,W,C))] * batch_size
    labels = [np.zeros((H,W))]* batch_size
    for i in range(0,batch_size):
        rand = np.random.randint(0,samples)
        feat[i],labels[i] = augmentData(np.expand_dims(X[rand],axis=0),np.expand_dims(Y[rand],axis=0))
    yield (feat,labels)

    
def validationGenerator(X,Y,batch_size):
    samples = X.shape[0]
    while True:
        random_im = np.random.choice(range(0,samples),batch_size)
        yield X[random_im],np.expand_dims(Y[random_im],axis=-1)
