"""

Script with utility functions related to the submission: RLE encoding and submission preparation

"""

from skimage.morphology import label
from skimage.transform import resize
import pandas as pd
import numpy as np

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    #lab_img = x
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
        
        
def prepareSubmission_new(test_names,predictions,submission_filename):       
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_names[0:]):
        print('Resizing back: '+str(n))
        rle = list(prob_to_rles(predictions[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))   
    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(submission_filename, index=False)
        
def prepareSubmission(test_names,predictions,original_sizes,submission_filename):
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_names[0:]):
        print('Resizing back: '+str(n))
        final_predictions = resize(predictions[n],original_sizes[n],mode='constant',preserve_range=True)
        rle = list(prob_to_rles(final_predictions))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))    
    
    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(submission_filename, index=False)
    
def unique(list1):
    # intilize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
    
def searchAppearence(l,w):
    for i in range(0,len(l)):
        if l[i] == w:
            return True
        else:
            pass
    return False
    
def findNonAppeared(list_all,list_some):
    non_app = []
    for i in range(0,len(list_all)):
        if searchAppearence(list_some,list_all[i]) == True:
            pass
        else:
            non_app = non_app + [list_all[i]]
    return non_app
    
