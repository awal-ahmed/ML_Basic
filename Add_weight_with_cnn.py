# This file is made to adjust class inbalance in semantic segmentation
import numpy as np
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

ONEHOT_MASK = '/content/drive/My Drive/Machine Learning/iccv09 3 class/mask/mask'  #path to the mask file
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256




# Adjust weight
loss_weights = {
    0: 0,
    1: 0,
    2: 0
}
mask_files = os.listdir(ONEHOT_MASK)
for mf in tqdm(mask_files):
    mask_img = cv2.imread(os.path.join(ONEHOT_MASK, mf))
    classes = tf.argmax(mask_img, axis=-1).numpy()
    class_counts = np.unique(classes, return_counts=True)
    
    for c in range(len(class_counts[0])):
        loss_weights[class_counts[0][c]] += class_counts[1][c]

print(loss_weights)

total = sum(loss_weights.values())
for cl, v in loss_weights.items():
    # do inverse
    loss_weights[cl] = total / (v*3)
    
loss_weights

w = [[loss_weights[0], loss_weights[1], loss_weights[2]]] * IMAGE_WIDTH
h = [w] * IMAGE_HEIGHT
loss_mod = np.array(h)




# Add your model here 
input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

inputs = tf.keras.Input(input_shape[-3:])

results = tf.keras.layers.Conv2D(48, kernel_size = (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(inputs)

output1 = tf.keras.layers.Conv2D(3, kernel_size = (1,1), padding = 'same', activation = 'softmax')(results)

model = tf.keras.Model(inputs = [inputs], outputs = [output1])


# Model ends

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              loss_weights=loss_mod)