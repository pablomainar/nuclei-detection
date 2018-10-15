import numpy as np
import os
from skimage.io import imread
from skimage.color import gray2rgb
from utils import *
from keras.models import Model, load_model

# Constant variables
TEST_PATH = './stage1_test/'
MODEL_PATH = './models/'

# Load the testing data
test_names = os.listdir(TEST_PATH)
test_number = len(test_names) # Change this to a smaller number to reduce the number of images

# Load model
model = load_model(os.path.join(MODEL_PATH,'final_model.h5'))

# Iterate over all images
for i in range(0,test_number):
    print('Doing test image number '+str(i+1)+' out of '+str(test_number))
    # Read image
    im_path = os.path.join(TEST_PATH, test_names[i], 'images', test_names[i] + '.png')
    im = imread(im_path, as_grey=False)
    if im.ndim < 3:
        im = gray2rgb(im)
    else:
        im = im[:,:,:3]

    # Add padding to have all dimensions multiple of 16
    X_test,pads_h,pads_w = pad_test(im)

    # Process image through net
    temp_lab, temp_cont = model.predict(np.expand_dims(X_test,axis=0),verbose=1)

    # Un pad the images for visualization
    Y_lab_test = unpad_test(temp_lab[0,:,:,0],pads_h,pads_w)
    Y_cont_test = unpad_test(temp_cont[0,:,:,0],pads_h,pads_w)
    X_test = unpad_test(X_test,pads_h,pads_w)

    # Build final prediction using both outputs of the net
    Y = (Y_lab_test > 0.5) & (Y_cont_test < 0.5)
    Y = Y.astype(np.int16)

    # See the result
    plt.figure(1)
    plt.subplot(121)
    plt.title('Original')
    plt.imshow(X_test)
    plt.subplot(122)
    plt.title('Segmentation')
    plt.imshow(Y)
    plt.show()
