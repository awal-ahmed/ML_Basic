# Copy code from All_model_run.py and replace model part with this code.

import numpy as np
import time
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Convolution2D, BatchNormalization, ReLU, LeakyReLU, Add, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, AveragePooling2D, UpSampling2D


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_CLASSES = 9

def convolutional_block(input_tensor, filters, block_identifier):
    # Dilated convolution block
    block_name = 'block_' + str(block_identifier) + '_'
    filter1, filter2, filter3 = filters
    skip_connection = input_tensor

    # Block A
    input_tensor = Convolution2D(filters=filter1, kernel_size=(1, 1), dilation_rate=(1, 1),
                      padding='same', kernel_initializer='he_normal', name=block_name + 'a')(input_tensor)
    input_tensor = BatchNormalization(name=block_name + 'batch_norm_a')(input_tensor)
    input_tensor = LeakyReLU(alpha=0.2, name=block_name + 'leakyrelu_a')(input_tensor)

    # Block B
    input_tensor = Convolution2D(filters=filter2, kernel_size=(3, 3), dilation_rate=(2, 2),
                      padding='same', kernel_initializer='he_normal', name=block_name + 'b')(input_tensor)
    input_tensor = BatchNormalization(name=block_name + 'batch_norm_b')(input_tensor)
    input_tensor = LeakyReLU(alpha=0.2, name=block_name + 'leakyrelu_b')(input_tensor)

    # Block C
    input_tensor = Convolution2D(filters=filter3, kernel_size=(1, 1), dilation_rate=(1, 1),
                      padding='same', kernel_initializer='he_normal', name=block_name + 'c')(input_tensor)
    input_tensor = BatchNormalization(name=block_name + 'batch_norm_c')(input_tensor)

    # Skip convolutional block for residual
    skip_connection = Convolution2D(filters=filter3, kernel_size=(3, 3), padding='same', name=block_name + 'skip_conv')(skip_connection)
    skip_connection = BatchNormalization(name=block_name + 'batch_norm_skip_conv')(skip_connection)

    # Block C + Skip Convolution
    input_tensor = Add(name=block_name + 'add')([input_tensor, skip_connection])
    input_tensor = ReLU(name=block_name + 'relu')(input_tensor)
    return input_tensor

def base_convolutional_block(input_layer):
    # Base convolutional block to obtain input image feature maps

    # Base Block 1
    base_result = convolutional_block(input_layer, [32, 32, 64], '1')

    # Base Block 2
    base_result = convolutional_block(base_result, [64, 64, 128], '2')

    # Base Block 3
    base_result = convolutional_block(base_result, [128, 128, 256], '3')

    return base_result

def pyramid_pooling_module(input_layer):
    # Pyramid pooling module
    base_result = base_convolutional_block(input_layer)

    # Red Pixel Pooling
    red_result = GlobalAveragePooling2D(name='red_pool')(base_result)
    red_result = tf.keras.layers.Reshape((1, 1, 256))(red_result)
    red_result = Convolution2D(filters=64, kernel_size=(1, 1), name='red_1_by_1')(red_result)
    red_result = UpSampling2D(size=256, interpolation='bilinear', name='red_upsampling')(red_result)

    # Yellow Pixel Pooling
    yellow_result = AveragePooling2D(pool_size=(2, 2), name='yellow_pool')(base_result)
    yellow_result = Convolution2D(filters=64, kernel_size=(1, 1), name='yellow_1_by_1')(yellow_result)
    yellow_result = UpSampling2D(size=2, interpolation='bilinear', name='yellow_upsampling')(yellow_result)

    # Blue Pixel Pooling
    blue_result = AveragePooling2D(pool_size=(4, 4), name='blue_pool')(base_result)
    blue_result = Convolution2D(filters=64, kernel_size=(1, 1), name='blue_1_by_1')(blue_result)
    blue_result = UpSampling2D(size=4, interpolation='bilinear', name='blue_upsampling')(blue_result)

    # Green Pixel Pooling
    green_result = AveragePooling2D(pool_size=(8, 8), name='green_pool')(base_result)
    green_result = Convolution2D(filters=64, kernel_size=(1, 1), name='green_1_by_1')(green_result)
    green_result = UpSampling2D(size=8, interpolation='bilinear', name='green_upsampling')(green_result)

    # Final Pyramid Pooling
    return tf.keras.layers.concatenate([base_result, red_result, yellow_result, blue_result, green_result])

def pyramid_based_conv(input_layer):
    result = pyramid_pooling_module(input_layer)
    result = Convolution2D(filters=num_classes, kernel_size=3, padding='same', name='last_conv_3_by_3')(result)
    result = BatchNormalization(name='last_conv_3_by_3_batch_norm')(result)
    result = Activation('sigmoid', name='last_conv_relu')(result)
    # result = tf.keras.layers.Flatten(name='last_conv_flatten')(result)
    return result

input_layer = tf.keras.Input(shape=((IMAGE_WIDTH, IMAGE_HEIGHT, 3)), name='input')
output_layer = pyramid_based_conv(input_layer)
model = Model(input_layer, output_layer)
