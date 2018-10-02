# 10/1/18

# CNN
# using tf.keras

# dataset: MNIST
# training set size: 60k
# test set size: 10k
# 28x28x1
# 10 class labels (0-9)

# Notes:
#   - bumping number of neurons in Dense up to 1024 dropped accuracy to ~10%
################################################################################
# IMPORTs
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import os
from datetime import datetime

################################################################################
print("TF version: {}".format(tf.__version__))
print("GPU available: {}".format(tf.test.is_gpu_available()))

################################################################################
# parameters
NUM_EPOCHS = 15
BATCH_SIZE = 128

################################################################################
# input image dimensions
img_rows, img_cols = 28, 28
num_channels = 1

# output dimensions
num_classes = 10

################################################################################
# load dataset
(training_images, training_labels), (testing_images, testing_labels) = tf.keras.datasets.mnist.load_data()

# shaping
input_shape = (img_rows, img_cols, num_channels)

# reshape images into shape=(-1, 28, 28, 1)
training_images = training_images.reshape(-1, img_rows, img_cols, num_channels)
testing_images = testing_images.reshape(-1, img_rows, img_cols, num_channels)

################################################################################
# build model
model = Sequential()

model.add(Conv2D(
    filters=32,
    kernel_size=[5, 5],
    input_shape=input_shape,
    activation=relu
))
model.add(MaxPool2D(
    pool_size=[2, 2],
    strides=2
))
model.add(Conv2D(
    filters=64,
    kernel_size=[5, 5],
    activation=relu
))
model.add(MaxPool2D(
    pool_size=[2, 2],
    strides=2
))
model.add(Flatten())
model.add(Dense(
    units=128,
    activation=relu
))
model.add(Dropout(0.25))
model.add(Dense(
    units=num_classes,
    activation=softmax
))

# configure model for training
model.compile(
    loss=sparse_categorical_crossentropy,
    optimizer=Adam(),
    metrics=["accuracy"]
)

model.summary()

################################################################################
# callbacks -> Tensorboard, Save weights
dir = os.path.join(os.getcwd(), datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
history_file = dir + "\cnn_tf_keras_mnist.h5"
save_callback = ModelCheckpoint(filepath=history_file, verbose=1)
tb_callback = TensorBoard(log_dir=dir)

# train model
model.fit(
    x=training_images,
    y=training_labels,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(testing_images, testing_labels),
    callbacks=[save_callback, tb_callback],
    verbose=1
)

# model.save(history_file)

# predictions
loss, accuracy = model.evaluate(
    x=testing_images,
    y=testing_labels,
    verbose=0
)
print("\n##########")
print("Loss: {}".format(loss))
print("Accuracy: {}".format(accuracy))
print("\n##########")
