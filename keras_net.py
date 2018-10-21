"""
Created on Mon Jun  4 19:44:58 2018

This scripts contains the function to create the deep nets. 

@author: pablo
"""
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.utils.data_utils import get_file

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# U-Net with one branch for segmentation.
def create_unet(H,W,C=3):
    # CNN architecture
    # Build U-Net model
    #inputs = Input((H, W, C))
    inputs = Input(shape=(None,None,C))
    s = Lambda(lambda x: x / 255.0) (inputs)
    
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)
    
    u6 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)
    
    u7 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)
    
    u8 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)
    
    u9 = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
    
    
# U-Net with two branches: one for segmentation andthe other for the contours
def create_unet_twoOutputs(C=3):
    inputs = Input(shape=(None,None,C))
    s = Lambda(lambda x: x / 255.0) (inputs)
    
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)
    
    u6c = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (c5)
    u6c = concatenate([u6c,c4])
    c6c = Conv2D(64, (3, 3), activation='relu', padding='same') (u6c)
    c6c = Conv2D(64, (3, 3), activation='relu', padding='same') (c6c)
    
    u7c = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same') (c6c)
    u7c = concatenate([u7c, c3])
    c7c = Conv2D(32, (3, 3), activation='relu', padding='same') (u7c)
    c7c = Conv2D(32, (3, 3), activation='relu', padding='same') (c7c)
    
    u8c = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same') (c7c)
    u8c = concatenate([u8c, c2])
    c8c = Conv2D(16, (3, 3), activation='relu', padding='same') (u8c)
    c8c = Conv2D(16, (3, 3), activation='relu', padding='same') (c8c)
    
    u9c = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same') (c8c)
    u9c = concatenate([u9c, c1], axis=3)
    c9c = Conv2D(8, (3, 3), activation='relu', padding='same') (u9c)
    c9c = Conv2D(8, (3, 3), activation='relu', padding='same') (c9c)
   
    contours = Conv2D(1, (1, 1), activation='sigmoid',name='contours') (c9c)
   
    u6l = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (c5)
    u6l = concatenate([u6l,c4])
    c6l = Conv2D(64, (3, 3), activation='relu', padding='same') (u6l)
    c6l = Conv2D(64, (3, 3), activation='relu', padding='same') (c6l)
    
    u7l = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same') (c6l)
    u7l = concatenate([u7l, c3])
    c7l = Conv2D(32, (3, 3), activation='relu', padding='same') (u7l)
    c7l = Conv2D(32, (3, 3), activation='relu', padding='same') (c7l)
    
    u8l = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same') (c7l)
    u8l = concatenate([u8l, c2])
    c8l = Conv2D(16, (3, 3), activation='relu', padding='same') (u8l)
    c8l = Conv2D(16, (3, 3), activation='relu', padding='same') (c8l)
    
    u9l = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same') (c8l)
    u9l = concatenate([u9l, c1], axis=3)
    c9l = Conv2D(8, (3, 3), activation='relu', padding='same') (u9l)
    c9l = Conv2D(8, (3, 3), activation='relu', padding='same') (c9l)


    labels = Conv2D(1, (1, 1), activation='sigmoid',name='labels') (c9l)
    
    model = Model(inputs=[inputs], outputs=[labels,contours])
    return model
    

# U-Net with VGG16 structure and initialization. It is too big for my laptop :(
def create_unet_vgg16(C=3):
    inputs = Input(shape=(None,None,C))
    s = Lambda(lambda x: x / 255.0) (inputs)
    
    # Block 1 down
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(s)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(c1)
    p1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(c1)
    
    # Block 2 down
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(c2)
    p2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(c2)
    
    # Block 3 down
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(c3)
    p3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(c3)
    
    # Block 4 down
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(c4)
    p4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(c4)

    # Block 5 down
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(c5)
    p5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(c5)
    
    model_vgg = Model(inputs=[inputs],outputs=[p5])
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',WEIGHTS_PATH_NO_TOP,cache_subdir='models')
    model_vgg.load_weights(weights_path)
    
    # Bottom of the U block
    c6 = Conv2D(512, (3,3), activation='relu', padding='same')(model_vgg.layers[-1].output)
    c6 = Conv2D(512, (3,3), activation='relu', padding='same')(c6)
    
    # Block 5 up
    u6 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same') (c6)
    u6 = concatenate([u6, model_vgg.layers[-2].output])
    c7 = Conv2D(512, (3,3), activation='relu', padding='same')(u6)
    c7 = Conv2D(512, (3,3), activation='relu', padding='same')(c7)
    c7 = Conv2D(512, (3,3), activation='relu', padding='same')(c7)
    
    # Block 4 up
    u7 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same') (c7)
    u7 = concatenate([u7,model_vgg.layers[-6].output])
    c8 = Conv2D(512, (3,3), activation='relu', padding='same')(u7)
    c8 = Conv2D(512, (3,3), activation='relu', padding='same')(c8)
    c8 = Conv2D(512, (3,3), activation='relu', padding='same')(c8)
    
    # Block 3 up
    u8 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same') (c8)
    u8 = concatenate([u8,model_vgg.layers[-10].output])
    c9 = Conv2D(256, (3,3), activation='relu', padding='same')(u8)
    c9 = Conv2D(256, (3,3), activation='relu', padding='same')(c9)
    c9 = Conv2D(256, (3,3), activation='relu', padding='same')(c9)
    
    # Block 2 up
    u9 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same') (c9)
    u9 = concatenate([u9,model_vgg.layers[-14].output])
    c10 = Conv2D(128, (3,3), activation='relu', padding='same')(u9)
    c10 = Conv2D(128, (3,3), activation='relu', padding='same')(c10)
    
    # Block 1 up
    u10 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same') (c10)
    u10 = concatenate([u10,model_vgg.layers[-17].output])
    c11 = Conv2D(64, (3,3), activation='relu', padding='same')(u10)
    c11 = Conv2D(64, (3,3), activation='relu', padding='same')(c11)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c11)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
