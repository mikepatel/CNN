########################################################
# CNN simple model

# dataset: cifar10
# 10 categories / class labels

# using keras

# 8/6/18 test accuracy is 71.27%
########################################################


# IMPORTs
from __future__ import print_function

import keras  # open source NN library
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator  # generate batches of tensor image data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
#from keras import backend as K

#import tensorflow as tf
from tensorflow.python.client import device_lib

import os

# GLOBALs
batch_size = 32
num_classes = 10  # cifar10
epochs = 100  # number of iterations on a dataset
data_augmentation = True  # increase number of data points / data images
# data augmentation allows model to re-use images with modifications (rotations, translations)...
# ...to pad up dataset when have limited amount of data
num_predictions = 20

save_dir = os.path.join(os.getcwd(), "saved_models")
model_name = "keras_cifar10_trained_model.h5"

# check if using GPU
#print(K.tensorflow_backend._get_available_gpus())
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print("Check if using gpu")
print(device_lib.list_local_devices())
#print(tf.Session(config=tf.ConfigProto(log_device_placement=True)))

# Data sets (training and test)
(train_feat, train_label), (test_feat, test_label) = cifar10.load_data()
print("training features shape: ", train_feat.shape)

# convert class vectors to binary class metrics
# vector (1 column) of class labels -> matrix with "num_classes" number of columns (1 for each category)
# vector -> matrix
train_label = keras.utils.to_categorical(train_label, num_classes)
test_label = keras.utils.to_categorical(test_label, num_classes)

# Model = structure to organize layers
model = Sequential()

# Conv2D -> spatial convolution over images
model.add(Conv2D(32,
                 (3, 3),
                 padding="same",
                 input_shape=train_feat.shape[1:]))
model.add(Activation("relu"))

model.add(Conv2D(32,
                 (3, 3)))
model.add(Activation("relu"))

model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # help prevent overfitting by randomly setting fraction rate of input units to 0

model.add(Conv2D(64,
                 (3, 3),
                 padding="same"))
model.add(Activation("relu"))

model.add(Conv2D(64,
                 (3, 3)))
model.add(Activation("relu"))

model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation("softmax"))

# RMSprop optimizer
optimizer = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# training
model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

train_feat = train_feat.astype("float32")
test_feat = test_feat.astype("float32")
train_feat /= 255
test_feat /= 255

if not data_augmentation:
    print("Not using data augmentation")
    model.fit(train_feat,
              train_label,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(test_feat, test_label),
              shuffle=True)
else:
    print("Using real-time data augmentation")
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=0,  # randomly rotate images (in degrees)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total width)
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode="nearest",  # set mode for filling points outside input boundaries
        cval=0.,  # value used for fill_mode="constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0
    )

    datagen.fit(train_feat)

    model.fit_generator(
        datagen.flow(train_feat, train_label,batch_size=batch_size),
        epochs=epochs,
        validation_data=(test_feat, test_label),
        workers=4
    )

# save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print("Saved trained model at %s " % model_path)

# Evaluate model / score trained model
scores = model.evaluate(test_feat, test_label, verbose=1)
print("Test loss: ", scores[0])
print("Test accuracy", scores[1])
