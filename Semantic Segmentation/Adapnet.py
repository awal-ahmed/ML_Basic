# Copy code from All_model_run.py and replace model part with this code.


import tensorflow as tf
import numpy as np

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_CLASSES = 9


def Upsampling(inputs,scale):
    return tf.keras.layers.UpSampling2D(scale)(inputs)

def ConvBlock(inputs, n_filters, kernel_size=[3, 3], stride=1):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """
    
    net = tf.keras.layers.BatchNormalization()(inputs);
    net = tf.keras.layers.ReLU()(net);
    net = tf.keras.layers.Conv2D(n_filters, kernel_size, stride=stride, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(net);
    
    return net

def ResNetBlock_1(inputs, filters_1, filters_2):
    net = tf.keras.layers.BatchNormalization()(inputs);
    net = tf.keras.layers.ReLU()(net);
    net = tf.keras.layers.Conv2D(filters_1, [1, 1],  padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(net);

    net = tf.keras.layers.BatchNormalization()(inputs);
    net = tf.keras.layers.ReLU()(net);
    net = tf.keras.layers.Conv2D(filters_1, [3, 3],  padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(net);


    net = tf.keras.layers.BatchNormalization()(inputs);
    net = tf.keras.layers.ReLU()(net);
    net = tf.keras.layers.Conv2D(filters_2, [1, 1],  padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(net);

    net = tf.add(inputs, net)

    return net

def ResNetBlock_2(inputs, filters_1, filters_2, s=1):
    
    net_1 = tf.keras.layers.BatchNormalization()(inputs);
    net_1 = tf.keras.layers.ReLU()(net_1);
    net_1 = tf.keras.layers.Conv2D(filters_1, [1, 1], stride=s, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(net_1);
    

    net_1 = tf.keras.layers.BatchNormalization()(net_1);
    net_1 = tf.keras.layers.ReLU()(net_1);
    net_1 = tf.keras.layers.Conv2D(filters_1, [3, 3], padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(net_1);
    
    
    net_1 = tf.keras.layers.BatchNormalization()(net_1);
    net_1 = tf.keras.layers.ReLU()(net_1);
    net_1 = tf.keras.layers.Conv2D(filters_2, [1, 1], padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(net_1);
    
    net_2 = tf.keras.layers.BatchNormalization()(inputs);
    net_2 = tf.keras.layers.ReLU()(net_2);
    net_2 = tf.keras.layers.Conv2D(filters_2, [1, 1], stride=s, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(net_2);
    
    net = tf.add(net_1, net_2)

    return net


def MultiscaleBlock_1(inputs, filters_1, filters_2, filters_3, p, d):
    net = tf.keras.layers.BatchNormalization()(inputs);
    net = tf.keras.layers.ReLU()(net);
    net = tf.keras.layers.Conv2D(filters_1, [1, 1],  padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(net);
    
    
    scale_1 = tf.keras.layers.BatchNormalization()(net);
    scale_1 = tf.keras.layers.ReLU()(scale_1);
    scale_1 = tf.keras.layers.Conv2D(filters_3 // 2, [3, 3], rate=p, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(scale_1);
    
    scale_2 = tf.keras.layers.BatchNormalization()(net);
    scale_2 = tf.keras.layers.ReLU()(scale_2);
    scale_2 = tf.keras.layers.Conv2D(filters_3 // 2, [3, 3], rate=d, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(scale_2);
    
    
    net = tf.concat((scale_1, scale_2), axis=-1)

    net = tf.keras.layers.BatchNormalization()(net);
    net = tf.keras.layers.ReLU()(net);
    net = tf.keras.layers.Conv2D(filters_2, [1, 1],  padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(net);
    

    net = tf.add(inputs, net)

    return net


def MultiscaleBlock_2(inputs, filters_1, filters_2, filters_3, p, d):
    net_1 = tf.keras.layers.BatchNormalization()(inputs);
    net_1 = tf.keras.layers.ReLU()(net_1);
    net_1 = tf.keras.layers.Conv2D(filters_1, [1, 1],  padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(net_1);
    
    
    scale_1 = tf.keras.layers.BatchNormalization()(net_1);
    scale_1 = tf.keras.layers.ReLU()(scale_1);
    scale_1 = tf.keras.layers.Conv2D(filters_3 // 2, [3, 3], rate=p, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(scale_1);
    
    scale_2 = tf.keras.layers.BatchNormalization()(net_1);
    scale_2 = tf.keras.layers.ReLU()(scale_2);
    scale_2 = tf.keras.layers.Conv2D(filters_3 // 2, [3, 3], rate=d, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(scale_2);
    
    net_1 = tf.concat((scale_1, scale_2), axis=-1)
    
    
    net_1 = tf.keras.layers.BatchNormalization()(net_1);
    net_1 = tf.keras.layers.ReLU()(net_1);
    net_1 = tf.keras.layers.Conv2D(filters_2, [1, 1],  padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(net_1);
    
    net_2 = tf.keras.layers.BatchNormalization()(inputs);
    net_2 = tf.keras.layers.ReLU()(net_2);
    net_2 = tf.keras.layers.Conv2D(filters_2, [1, 1],  padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(net_2);
    

    net = tf.add(net_1, net_2)

    return net

inputs = tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

net = ConvBlock(s, n_filters=64, kernel_size=[3, 3])
net = ConvBlock(net, n_filters=64, kernel_size=[7, 7], stride=2)
net = tf.keras.layers.MaxPooling2D((2,2), stride=[2, 2])(net)


net = ResNetBlock_2(net, filters_1=64, filters_2=256, s=1)
net = ResNetBlock_1(net, filters_1=64, filters_2=256)
net = ResNetBlock_1(net, filters_1=64, filters_2=256)

net = ResNetBlock_2(net, filters_1=128, filters_2=512, s=2)
net = ResNetBlock_1(net, filters_1=128, filters_2=512)
net = ResNetBlock_1(net, filters_1=128, filters_2=512)

skip_connection = ConvBlock(net, n_filters=12, kernel_size=[1, 1])


net = MultiscaleBlock_1(net, filters_1=128, filters_2=512, filters_3=64, p=1, d=2)

net = ResNetBlock_2(net, filters_1=256, filters_2=1024, s=2)
net = ResNetBlock_1(net, filters_1=256, filters_2=1024)
net = MultiscaleBlock_1(net, filters_1=256, filters_2=1024, filters_3=64, p=1, d=2)
net = MultiscaleBlock_1(net, filters_1=256, filters_2=1024, filters_3=64, p=1, d=4)
net = MultiscaleBlock_1(net, filters_1=256, filters_2=1024, filters_3=64, p=1, d=8)
net = MultiscaleBlock_1(net, filters_1=256, filters_2=1024, filters_3=64, p=1, d=16)

net = MultiscaleBlock_2(net, filters_1=512, filters_2=2048, filters_3=512, p=2, d=4)
net = MultiscaleBlock_1(net, filters_1=512, filters_2=2048, filters_3=512, p=2, d=8)
net = MultiscaleBlock_1(net, filters_1=512, filters_2=2048, filters_3=512, p=2, d=16)

net = ConvBlock(net, n_filters=12, kernel_size=[1, 1])
net = Upsampling(net, scale=2)


net = tf.add(skip_connection, net)

net = ConvBlock(net, n_filters=6, kernel_size=[1, 1])
net = Upsampling(net, scale=8)


net = tf.keras.layers.conv2d(net, NUM_CLASSES, [1, 1], activation_fn='softmax', scope='logits')



model = tf.keras.Model(inputs = [inputs], outputs = [net])