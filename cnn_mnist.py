# TF Layers: CNN tutorial
# use MNIST data set to classify handwritten digits (0-9)
# training set size: 60,000
# test set size: 10,000
# 28 x 28 pixel images
# monochrome images (greyscale)

###################################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


# tf.logging.set_verbosity(tf.logging.INFO)

###################################################################################
# build CNN model
def cnn_model_fn(features, labels, mode):  # model function for CNN
    # INPUT Layer
    '''
    Reshape X to 4-D tensor: [batch_size, width, height, channels]
    MNIST images are 28x28 pixels w/ 1 color
    '''
    input_layer = tf.reshape(features["x"],
                             [-1, 28, 28, 1])  # -1 batch size signifies this dimension is dynamically computed

    # CONVOLUTIONAL Layer #1
    '''
    - computes 32 features using 5x5 filter w/ ReLU activation
    - padding is added to preserve width and height
    - Input tensor shape: [batch_size, 28, 28, 1]
      Output tensor shape: [batch_size, 28, 28, 32]
    '''
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # POOLING Layer #1
    '''
    - first max pooling layer w/ 2x2 filter and stride=2
    - Input tensor shape: [batch_size, 28, 28, 32]
      Output tensor shape: [batch_size, 14, 14, 32]
    '''
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # CONVOLUTIONAL Layer #2
    '''
    - computes 64 features using 5x5 filter w/ ReLU activation
    - padding is added to preserve width and height
    - Input tensor shape: [batch_size, 14, 14, 32]
      Output tensor shape: [batch_size, 14, 14, 64]
    '''
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # POOLING Layer #2
    '''
    - second max pooling layer w/ 2x2 filter and stride=2
    - Input tensor shape: [batch_size, 14, 14, 64]
      Output tensor shape: [batch_size, 7, 7, 64]
    '''
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # FLATTEN tensor into a batch of vectors
    '''
    - Input tensor shape: [batch_size, 7, 7, 64]
      Output tensor shape: [batch_size, 7*7*64]
    '''
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # DENSE Layer
    '''
    - densely connected layer w/ 1024 neurons
    - Input tensor shape: [batch_size, 7*7*64]
      Output tensor shape: [batch_size, 1024]
    '''
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # DROPOUT
    # 0.6 probability that element will be kept
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)  # TRAIN mode

    # LOGITS Layer
    '''
    - Input tensor shape: [batch_size, 1024]
      Output tensor shape: [batch_size, 10]
    '''
    logits = tf.layers.dense(inputs=dropout, units=10)

    # PREDICTIONs
    # generate predictions (for PREDICT and EVAL mode)
    # predictions have 2 formats: predicted class AND probabilities for each target class
    # argmax => majority voting
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),  # predicted class based on highest raw value, index of 1
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    # add 'softmax_tensor' to the graph, get probabilities from softmax
    }

    if (mode == tf.estimator.ModeKeys.PREDICT):
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # LOSSes
    # for TRAIN and EVAL modes
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # TRAINING OPERATION
    # TRAIN mode
    # learning optimizations
    # stochastic gradient descent (SGD) as optimization algorithm
    if (mode == tf.estimator.ModeKeys.TRAIN):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # EVAL metrics
    # EVAL mode
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


###################################################################################
# main()
def main(unused_argv):
    # load training and evaluation data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # create the estimator
    '''
    Estimator = TF class that performs high-level model training, evaluation and infererence
    '''
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    # set up logging for predictions
    # track progress during long training
    # log the values in the 'softmax' tensor w/ label 'probabilities'
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook]
    )

    print("trained")

    # evaluate the model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )
    eval_results = mnist_classifier.evaluate(
        input_fn=eval_input_fn
    )

    print(eval_results)


if __name__ == "__main__":
    tf.app.run()