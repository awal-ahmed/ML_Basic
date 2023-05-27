import numpy as np
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras


np.random.seed(43)
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 16
NUM_CLASSES = 3 # Number of classes
IMG_PATH = '/content/drive/My Drive/train Image/' # Image root path / one folder above where image actually belongs
MASK_PATH = '/content/drive/My Drive/mask/'       # Mask root path / one folder above where mask actually belongs
SAVE_MODEL = '/content/drive/MyDrive/resnet50_without_wights.h5'    #Where you want to save your model and it's name


training_generation_args = dict(
     #width_shift_range=0.3,
     #height_shift_range=0.3,
    horizontal_flip=True,
    #vertical_flip=True,
    zoom_range=0.2,
    validation_split=0.1,
)
train_image_datagen = ImageDataGenerator(**training_generation_args)
train_label_datagen = ImageDataGenerator(**training_generation_args)

# data load
training_image_generator = train_image_datagen.flow_from_directory(
    IMG_PATH,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    class_mode=None,
    subset='training',
    batch_size=BATCH_SIZE,
    seed=1
)
training_label_generator = train_label_datagen.flow_from_directory(
    MASK_PATH,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    class_mode=None,
    subset='training',
    batch_size=BATCH_SIZE,
    seed=1
)


# validation data load
validation_image_generator = train_image_datagen.flow_from_directory(
    IMG_PATH,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    class_mode=None,
    subset='validation',
    batch_size=BATCH_SIZE,
    seed=1
)
validation_label_generator = train_label_datagen.flow_from_directory(
    MASK_PATH,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    class_mode=None,
    subset='validation',
    batch_size=BATCH_SIZE,
    seed=1
)

train_generator = zip(training_image_generator, training_label_generator)
validation_generator = zip(validation_image_generator, validation_label_generator)


# Change this model according to the requirement

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
# Model ends here

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


ACCURACY_THRESHOLD = 0.0
class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        global ACCURACY_THRESHOLD
        if(logs.get('val_accuracy') > ACCURACY_THRESHOLD ):
            ACCURACY_THRESHOLD = logs.get('val_accuracy')
            model.save(SAVE_MODEL)
            
callbacks = myCallback()


model_history = model.fit(train_generator,
                          epochs=100,
                          steps_per_epoch=training_image_generator.samples // BATCH_SIZE,
                          #shuffle=True,
                          validation_data=validation_generator,
                          validation_steps=validation_image_generator.samples // BATCH_SIZE,
                          callbacks=[callbacks])


model = keras.models.load_model(SAVE_MODEL)



# Individual image prediction score
img = cv2.imread(os.path.join("/content/drive/My Drive/1.jpg"))     # Pathe of a single image
img =  cv2.resize(img, (256, 256))
img=np.expand_dims(img, 0)
img = model.predict(img)
img =  np.squeeze(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img > .5
test1 = cv2.imread(os.path.join("/content/drive/My Drive/mask1.png"))  # Pathe for the mask of that image
test1 =  cv2.resize(test1, (256, 256))
test1 = test1/255
TP = tf.math.count_nonzero(img * test1)
TN = tf.math.count_nonzero((img - 1) * (test1 - 1))
FP = tf.math.count_nonzero(img * (test1 - 1))
FN = tf.math.count_nonzero((img - 1) * test1)
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
print("Recall: ", recall)
print("Precision: ", precision)
print("Accuracy: ", accuracy)
print("f1 score: ", f1)



# Full dataset prediction score
test_p = "/content/drive/My Drive/test Image/"      # Test image root path / one folder above where test image actually belongs
test_m = "/content/drive/My Drive/test one hot/"    # Test mask root path / one folder above where test mask actually belongs
number_of_test_images = 102


test= np.zeros((number_of_test_images, IMAGE_HEIGHT , IMAGE_WIDTH, 3), dtype= np.bool)
tst_fies = os.listdir(test_m)
tst_fies.sort()
for n, mf in tqdm(enumerate(tst_fies), total=len(tst_fies)):
  img = cv2.imread(os.path.join(test_m, mf) )
  img = cv2.resize(img, (256, 256))
  img = img / 255
  test[n] = img


acc = []
pre = []
rec = []
f_me = []
test_files = os.listdir(test_p)
test_files.sort()
for n, mf in tqdm(enumerate(test_files), total=len(test_files)):
  img = cv2.imread(os.path.join(test_p, mf) )
  img =  cv2.resize(img, (256, 256))
  img=np.expand_dims(img, 0)
  img = model.predict(img)
  img =  np.squeeze(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = img > .5
  test1 = test[n]
  TP = tf.math.count_nonzero(img * test1)
  TN = tf.math.count_nonzero((img - 1) * (test1 - 1))
  FP = tf.math.count_nonzero(img * (test1 - 1))
  FN = tf.math.count_nonzero((img - 1) * test1)
  accuracy = (TP + TN) / (TP + TN + FP + FN)
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  f1 = 2 * precision * recall / (precision + recall)
  acc.append(accuracy)
  pre.append(precision)
  rec.append(recall)
  f_me.append(f1)


print("Mean")
print("Recall: ", sum(rec)/number_of_test_images)
print("Precision: ", sum(pre)/number_of_test_images)
print("Accuracy: ", sum(acc)/number_of_test_images)
print("f1 score: ", sum(f_me)/number_of_test_images)

print("SD")
print("Recall: ", np.std(rec))
print("Precision: ", np.std(pre))
print("Accuracy: ", np.std(acc))
print("f1 score: ", np.std(f_me))