# Tutorial using Keras
# CNN architecture
# dataset = MNIST

# IMPORTs
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# parameters
num_epochs = 5
num_class_labels = 10
batch_size = 128

# input image dimensions
img_rows, img_cols = 28, 28

# load data and split into training and test sets
(train_feat, train_label), (test_feat, test_label) = mnist.load_data()

#
if K.image_data_format() == "channels_first":
    train_feat = train_feat.reshape(train_feat.shape[0], 1, img_rows, img_cols)
    test_feat = test_feat.reshape(test_feat.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_feat = train_feat.reshape(train_feat.shape[0], img_rows, img_cols, 1)
    test_feat = test_feat.reshape(test_feat.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#
train_feat = train_feat.astype("float32")
test_feat = test_feat.astype("float32")
train_feat /= 255
test_feat /= 255
print("train_feat shape: ", train_feat.shape)
print(train_feat.shape[0], "train samples")
print(test_feat.shape[0], "test samples")

# convert class vectors to binary class matrices
train_label = keras.utils.to_categorical(train_label, num_class_labels)
test_label = keras.utils.to_categorical(test_label, num_class_labels)

# build model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_class_labels, activation="softmax"))

# compile model (configure learning process)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=["accuracy"])

# train model
model.fit(train_feat, train_label,
          batch_size=batch_size,
          epochs=num_epochs,
          verbose=1,
          validation_data=(test_feat, test_label))

#
score = model.evaluate(test_feat, test_label, verbose=0)
print("Total loss: ", score[0])
print("Test accuracy: ", score[1])
