# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 18:38:36 2018

@author: pablo
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize
import skimage.segmentation
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max


def watershed_segmentation(Y):
    Y_new = Y[:]
    for i in range(0,Y.shape[0]):
        im = Y[i]
        distance = ndi.distance_transform_edt(im)
        local_maxi = peak_local_max(distance,indices=False,footprint=np.ones((3,3)),labels=im)
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-distance,markers,mask=im)
        Y_new[i] = labels.astype(np.int16)
    return Y_new


def visualize(X,Y,index=None):
    if index is None:
        index = np.random.randint(0,X.shape[0])
    if len(X.shape) == 4:
        im = X[index]
        lab = np.squeeze(Y[index])
    else:
        im = X
        lab = Y
    print('Image number '+str(index))
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(im)
    plt.subplot(122)
    plt.imshow(lab)
    plt.show()
    
    
#TO-DO implement more data augmentation ways
def augmentData(X,Y):
    Xaug = np.zeros(X.shape).astype(np.int16)
    Yaug = np.zeros(Y.shape).astype(np.int16)
    for i in range(0,X.shape[0]):
        action = np.random.choice(['identity','rotate','flip','gaussian','crop','rgb_shuffle','blur'])       
        if action == 'identity':
            Xaug[i] = X[i]
            Yaug[i] = Y[i]
        elif action == 'rotate':
            angle = np.random.choice([90,180,270])
            (h,w) = X[i].shape[:2]
            center = (h/2,w/2)
            M = cv2.getRotationMatrix2D(center,angle,1)
            Xaug[i] = cv2.warpAffine(X[i],M,(h,w)).astype(np.int16)
            Yaug[i] = cv2.warpAffine(Y[i],M,(h,w)).astype(np.int16)
        elif action == 'flip':
            flip = np.random.choice([0,1])
            Xaug[i] = cv2.flip(X[i],flip)
            Yaug[i] = np.squeeze(cv2.flip(Y[i],flip))
        elif action == 'gaussian':
            mean = 0
            var = 150
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,X[i].shape)
            gauss = gauss.astype(np.int16)
            Xaug[i] = X[i] + gauss
            Yaug[i] = Y[i]
        elif action == 'crop':
            (h,w) = X[i].shape[:2]
            crop = np.random.randint(160,240)
            corner = np.random.choice([0,1,2,3])
            if corner == 0:
                im = X[i,:crop,:crop,:]
                lab = Y[i,:crop,:crop]
            elif corner == 1:
                im = X[i,:crop,-crop:,:]
                lab = Y[i,:crop,-crop:]
            elif corner == 2:
                im = X[i,-crop:,:crop,:]
                lab = Y[i,-crop:,:crop]
            elif corner == 3:
                im = X[i,-crop:,-crop:,:]
                lab = Y[i,-crop:,-crop:]
            Xaug[i] = resize(im,(h,w,3),preserve_range=True,order=0).astype(np.int16)
            Yaug[i] = resize(lab,(h,w),preserve_range=True,order=0).astype(np.int16)
        elif action == 'rgb_shuffle':
            im = X[i]
            im = np.swapaxes(im,0,-1)
            np.random.shuffle(im)
            im = np.swapaxes(im,0,-1)
            Xaug[i] = im
            Yaug[i] = Y[i]
        elif action == 'blur':
            blur = np.random.choose(['average','gaussian'])
            kernel_size = np.random.randint(5,10)
            if blur == 'average':
                Xaug[i] = cv2.blur(X[i],(kernel_size,kernel_size))
            elif blur == 'gaussian':
                Xaug[i] = cv2.GaussianBlur(X[i],(kernel_size,kernel_size),0)
            Yaug[i] = Y[i]
    return Xaug.astype(np.int16),Yaug.astype(np.int16)
    
    

def augmentDataTwoOutputs(X,Ylab,Ycont):
    Xaug = np.zeros(X.shape).astype(np.int16)
    Ylabaug = np.zeros(Ylab.shape).astype(np.int16)
    Ycontaug = np.zeros(Ycont.shape).astype(np.int16)
    for i in range(0,X.shape[0]):
        action = np.random.choice(['identity','rotate','flip','gaussian','rgb_shuffle'])       
        if action == 'identity':
            Xaug[i] = X[i]
            Ylabaug[i] = Ylab[i]
            Ycontaug[i] = Ycont[i]
        elif action == 'rotate':
            angle = np.random.choice([90,180,270])
            (h,w) = X[i].shape[:2]
            center = (h/2,w/2)
            M = cv2.getRotationMatrix2D(center,angle,1)
            Xaug[i] = cv2.warpAffine(X[i],M,(h,w)).astype(np.int16)
            Ylabaug[i] = cv2.warpAffine(Ylab[i],M,(h,w)).astype(np.int16)
            Ycontaug[i] = cv2.warpAffine(Ycont[i],M,(h,w)).astype(np.int16)
        elif action == 'flip':
            flip = np.random.choice([0,1])
            Xaug[i] = cv2.flip(X[i],flip)
            Ylabaug[i] = np.squeeze(cv2.flip(Ylab[i],flip))
            Ycontaug[i] = np.squeeze(cv2.flip(Ycont[i],flip))
        elif action == 'gaussian':
            mean = 0
            var = 150
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,X[i].shape)
            gauss = gauss.astype(np.int16)
            Xaug[i] = X[i] + gauss
            Ylabaug[i] = Ylab[i]
            Ycontaug[i] = Ycont[i]
        elif action == 'rgb_shuffle':
            im = X[i]
            im = np.swapaxes(im,0,-1)
            np.random.shuffle(im)
            im = np.swapaxes(im,0,-1)
            Xaug[i] = im
            Ylabaug[i] = Ylab[i]
            Ycontaug[i] = Ycont[i]
    return Xaug.astype(np.int16),Ylabaug.astype(np.int16),Ycontaug.astype(np.int16)
    
    
    
    
def unpad_test(im,pad_h,pad_w):
    if (pad_h == 0) & (pad_w == 0):
        return im
    elif (pad_h != 0) & (pad_w == 0):
        return im[:-pad_h,:]
    elif (pad_h == 0) & (pad_w != 0):
        return im[:,:-pad_w]
    elif (pad_h != 0) & (pad_w != 0):
        return im[:-pad_h,:-pad_w]

def pad_test(im):
    h,w = im.shape[:2]
    if h % 16 != 0:
        pad_h = ((np.ceil(h/16.0) * 16) - h).astype(int)
    else:
        pad_h = 0
    if w % 16 != 0:
        pad_w = ((np.ceil(w/16.0) * 16) - w).astype(int)
    else:
        pad_w = 0
    
    if (pad_h == 0) & (pad_w == 0):
        return (im.astype(np.int16),0,0)
    else:
        new_im = np.zeros((h+pad_h,w+pad_w,3))
        new_im[:h,:w,:] = im
        return (new_im.astype(np.int16),pad_h,pad_w)
    
# Precision helper function
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp.astype(float), fp, fn

def computeIoU(X,Ytrue,Ypred,verbose=0,return_all_pred=False):
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
        
        y_pred, _, _ = skimage.segmentation.relabel_sequential(y_pred) # Relabel objects
        
        
        # Compute number of objects
        true_objects = len(np.unique(labels))
        pred_objects = len(np.unique(y_pred))
        if verbose == 1:
            print('Iter '+str(i)+': Number of true objects:'+str(true_objects))
            print('Iter '+str(i)+': Number of predicted objects:'+str(pred_objects))
        
        # Compute intersection between all objects
        intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]
        
        # Compute areas (needed for finding the union between all objects)
        area_true = np.histogram(labels, bins = true_objects)[0]
        area_pred = np.histogram(y_pred, bins = pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)
        
        # Compute union
        union = area_true + area_pred - intersection
        
        # Exclude background from the analysis
        intersection = intersection[1:,1:]
        union = union[1:,1:]
        union[union == 0] = 1e-9
        
        # Compute the intersection over union
        iou = intersection / union
        
        # Loop over IoU thresholds
        prec = []
        #print("Thresh\tTP\tFP\tFN\tPrec.")
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, iou)
            p = tp  / (tp + fp + fn)
            #print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
            prec.append(p)
        #print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
        all_prec.append(np.mean(prec))
        if verbose == 1:
            print('Iter '+str(i)+': Score: '+str(np.mean(prec)))
    if return_all_pred == False:
        return np.mean(all_prec)
    else:
        return all_prec
        
        
def doTTA(X):
    new_X = np.zeros((4,X.shape[0],X.shape[1],X.shape[2]))
    new_X[0] = X
    new_X[1] = cv2.flip(X,0)
    new_X[2] = cv2.flip(X,1)
    (h,w) = X.shape[:2]
    center = (w/2,h/2)
    M = cv2.getRotationMatrix2D(center,180,1)
    new_X[3] = cv2.warpAffine(X,M,(w,h)).astype(np.int16)
    return new_X.astype(np.int16)
    
def undoTTA(Y):
    new_Y = np.zeros(Y.shape)
    new_Y[0] = Y[0]
    new_Y[1] = cv2.flip(Y[1],0)
    new_Y[2] = cv2.flip(Y[2],1)
    (h,w) = Y.shape[1:3]
    center = (w/2,h/2)
    M = cv2.getRotationMatrix2D(center,180,1)
    new_Y[3] = cv2.warpAffine(Y[3],M,(w,h))
    new_Y = np.sum(new_Y,axis=0)
    return new_Y / 4.0
    