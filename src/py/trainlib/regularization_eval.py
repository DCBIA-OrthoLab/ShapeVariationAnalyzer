
# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 4
# ------------
# 
# Previously in `2_fullyconnected.ipynb` and `3_regularization.ipynb`, we trained fully connected networks to classify [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) characters.
# 
# The goal of this assignment is make the neural network convolutional.

# In[ ]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import argparse
import regularization_nn as nn
import os

print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Model file computed with regularization_train.py', required=True)
parser.add_argument('--sampleMesh', help='Evaluate an image sample in vtk format')
parser.add_argument('--out', help='Write output of evaluation', default="", type=str)
parser.add_argument('--pickle', help='Pickle file, check the script readImages to generate this file.')
parser.add_argument('--batch', help='Batch size for evaluation', default=64)

args = parser.parse_args()

pickle_file = args.pickle
outvariablesfilename = args.out
batch_size = args.batch
model = args.model
num_labels = 8

f = open(pickle_file, 'rb')
data = pickle.load(f)
valid_dataset = data["valid_dataset"]
valid_labels = data["valid_labels"]
test_dataset = data["test_dataset"]
test_labels = data["test_labels"]
f.close()

# Reformat into a TensorFlow-friendly shape:
# - convolutions need the image data formatted as a cube (width by height by #channels)
# - labels as float 1-hot encodings.

# In[ ]:

# in_depth = img_size[3] #zdim
# in_height = img_size[2] #ydim
# in_width = img_size[1] #xdim
# num_channels = img_size[0] #numchannels
# num_channels_labels = 1

# Reformat into a TensorFlow-friendly shape:
# - convolutions need the image data formatted as a cube (depth * width * height * channels)
# - We know that nrrd format 
# - labels as float 1-hot encodings.


def reformat(dataset, labels):
  dataset = dataset.reshape(dataset.shape[0], -1)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

size_features = valid_dataset.shape[1]

print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
print('batch_size', batch_size)


# Let's build a small network with two convolutional layers, followed by one fully connected layer. Convolutional networks are more expensive computationally, so we'll limit its depth and number of fully connected nodes.

# In[ ]:

#batch_size = 64
# patch_size = 8
# depth = 32
# depth2 = 64
# num_hidden = 256
# stride = [1, 1, 1, 1]

# def evaluate_accuracy(prediction, labels):    
#   accuracy = tf.reduce_sum(tf.squared_difference(prediction, labels))
#   return accuracy.eval()

graph = tf.Graph()

with graph.as_default():

# run inference on the input data
  x = tf.placeholder(tf.float32,shape=(batch_size, size_features))
  y_ = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  
  keep_prob = tf.placeholder(tf.float32)

  #tf_valid_dataset = tf.constant(valid_dataset)
  # tf_test_dataset = tf.constant(test_dataset)

  y_conv = nn.inference(x, size_features, num_labels, keep_prob, batch_size)

# calculate the loss from the results of inference and the labels
  #loss = nn.loss(y_conv, y_)

  accuracy_eval = nn.evaluation(y_conv, y_)

  #tf.summary.scalar(loss.op.name, loss)

  #intersection_sum, label_sum, example_sum, precision = nn.evaluation(y_conv, y_)

  #tf.summary.scalar ("Precision op", precision)

# setup the training operations
  #train_step = nn.training(loss, learning_rate, decay_steps, decay_rate)
  # setup the summary ops to use TensorBoard

  # setup the training operations
  #train_step = nn.training(loss, learning_rate, decay_steps, decay_rate)
  #train_step = nn.training(loss, learning_rate, decay_steps, decay_rate)

  # intersection_sum, label_sum, example_sum = evaluation(y_conv, y_)

  # valid_prediction = model(tf_valid_dataset)
  #cross_entropy = tf.reduce_sum(tf.squared_difference(y_conv, y_))

  #regularizers = tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
  #cross_entropy += 0.1 * regularizers

  #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
  #train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)  

  # accuracy = cross_entropy

  # valid_prediction = model(tf_valid_dataset)
  # evaluation(valid_prediction)
  # test_prediction = model(tf_test_dataset)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, model)


    for step in range(int(len(valid_dataset)/batch_size)):

      offset = (step * batch_size) % (valid_dataset.shape[0] - batch_size)
      batch_data = valid_dataset[offset:(offset + batch_size), :]
      batch_labels = valid_labels[offset:(offset + batch_size), :]

      accuracy = sess.run([accuracy_eval], feed_dict={x: batch_data, y_: batch_labels, keep_prob: 1})
      
      print('OUTPUT: Step %d: accuracy = %.3f' % (step, accuracy[0]))

    print('Evaluate test dataset') 
    for step in range(int(len(test_dataset))):

      offset = (step * batch_size) % (test_dataset.shape[0] - batch_size)
      batch_data = test_dataset[offset:(offset + batch_size), :]
      batch_labels = test_labels[offset:(offset + batch_size), :]

      accuracy = sess.run([accuracy_eval], feed_dict={x: batch_data, y_: batch_labels, keep_prob: 1})
      
      print('OUTPUT: Step %d: accuracy = %.3f' % (step, accuracy[0]))        

    #test_accuracy = evaluate_accuracy(test_prediction.eval(feed_dict={keep_prob: 1.0}), test_labels)
    #print("test accuracy %g"%test_accuracy)
    
  
  
