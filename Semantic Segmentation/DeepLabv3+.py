import tensorflow as tf

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_CLASSES = 3


def AtrousSpatialPyramidPooling(input_shape):

  inputs = tf.keras.Input(input_shape[-3:])
  results = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, [1,2], keepdims = True))(inputs)
  results = tf.keras.layers.Conv2D(256, kernel_size = (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(results)
  results = tf.keras.layers.BatchNormalization()(results)
  results = tf.keras.layers.ReLU()(results)
  pool = tf.keras.layers.UpSampling2D(size = (input_shape[-3] // results.shape[1], input_shape[-2] // results.shape[2]), interpolation = 'bilinear')(results)
  results = tf.keras.layers.Conv2D(256, kernel_size = (1,1), dilation_rate = 1, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(inputs)
  results = tf.keras.layers.BatchNormalization()(results)
  dilated_1 = tf.keras.layers.ReLU()(results)

  results = tf.keras.layers.Conv2D(256, kernel_size = (3,3), dilation_rate = 6, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(inputs)
  results = tf.keras.layers.BatchNormalization()(results)
  dilated_6 = tf.keras.layers.ReLU()(results)
  results = tf.keras.layers.Conv2D(256, kernel_size = (3,3), dilation_rate = 12, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(inputs)
  results = tf.keras.layers.BatchNormalization()(results)
  dilated_12 = tf.keras.layers.ReLU()(results)

  results = tf.keras.layers.Conv2D(256, kernel_size = (3,3), dilation_rate = 18, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(inputs)
  results = tf.keras.layers.BatchNormalization()(results)
  dilated_18 = tf.keras.layers.ReLU()(results)

  results = tf.keras.layers.Concatenate(axis = -1)([dilated_1, dilated_6, dilated_12, dilated_18, pool])
  results = tf.keras.layers.Conv2D(256, kernel_size = (1,1), dilation_rate = 1, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(results)
  results = tf.keras.layers.BatchNormalization()(results)
  results = tf.keras.layers.ReLU()(results)
  return tf.keras.Model(inputs = inputs, outputs = results)

input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

inputs = tf.keras.Input(input_shape[-3:])
resnet50 = tf.keras.applications.ResNet50(input_tensor = inputs, weights = 'imagenet', include_top = False)
results = resnet50.get_layer('conv4_block6_2_relu').output
results = AtrousSpatialPyramidPooling(results.shape[-3:])(results)
a = tf.keras.layers.UpSampling2D(size = (input_shape[-3] // 4 // results.shape[1], input_shape[-2] // 4 // results.shape[2]), interpolation = 'bilinear')(results)

results = resnet50.get_layer('conv2_block3_2_relu').output
results = tf.keras.layers.Conv2D(48, kernel_size = (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(results)
results = tf.keras.layers.BatchNormalization()(results)
b = tf.keras.layers.ReLU()(results)

results = tf.keras.layers.Concatenate(axis = -1)([a, b])
results = tf.keras.layers.Conv2D(256, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(results)
results = tf.keras.layers.BatchNormalization()(results)
results = tf.keras.layers.ReLU()(results)
results = tf.keras.layers.Conv2D(32, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(results)
results = tf.keras.layers.BatchNormalization()(results)
results = tf.keras.layers.ReLU()(results)
results = tf.keras.layers.UpSampling2D(size = (input_shape[-3] // results.shape[1], input_shape[-2] // results.shape[2]), interpolation = 'bilinear')(results)
output1 = tf.keras.layers.Conv2D(NUM_CLASSES, kernel_size = (1,1), padding = 'same', activation = 'softmax')(results)

model = tf.keras.Model(inputs = [inputs], outputs = [output1])