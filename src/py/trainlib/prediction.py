
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
import neuralNetwork as nn
import os
import vtk
import glob
import sys

sys.path.append('../generatelib')

import inputData

print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser(description='Shape Variation Analyzer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', help='Model file computed with train.py', required=True)

group = parser.add_mutually_exclusive_group()
group.add_argument('--sampleMesh', help='Evaluate an image sample in vtk format')
group.add_argument('--sampleDir', help='Evaluate a directory with vtk files')

parser.add_argument('--out', help='Write output of evaluation', default=None, type=str)
parser.add_argument('--feature_names', help='Extract the following features from the polydatas', nargs='+', default=["Normals", "Mean_Curvature", "distanceGroup"], type=str)
parser.add_argument('--num_labels', help='Number of labels', type=int, default=7)

args = parser.parse_args()

sampleMesh = args.sampleMesh
sampleDir = args.sampleDir
outfilename = args.out
model = args.model
feature_names = args.feature_names
num_labels = args.num_labels

inputdata = inputData.inputData()

if sampleMesh != None:
  valid_dataset = inputdata.load_features(sampleMesh)
  valid_dataset = valid_dataset.reshape(1, -1)
  batch_size = 1
else:           
  vtklist = glob.glob(os.path.join(sampleDir, "*.vtk"))
  valid_dataset = inputdata.load_features_classe(vtklist, feature_names=feature_names)
  valid_dataset = valid_dataset.reshape(valid_dataset.shape[0], -1)
  batch_size = valid_dataset.shape[0]

size_features = valid_dataset.shape[1]

print('Validation set', valid_dataset.shape)


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
  
  keep_prob = tf.placeholder(tf.float32)

  #tf_valid_dataset = tf.constant(valid_dataset)
  # tf_test_dataset = tf.constant(test_dataset)

  y_conv = nn.inference(x, size_features, num_labels, keep_prob, batch_size)
  predict = tf.argmax(y_conv, 1)

# calculate the loss from the results of inference and the labels
  #loss = nn.loss(y_conv, y_)

  #accuracy_eval = nn.evaluation(y_conv, y_)

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

    pred = sess.run([predict], feed_dict={x: valid_dataset, keep_prob: 1})
    
    if outfilename != None:
      with open(outfilename, "w") as outfile:
        for pr in pred[0]:
          outfile.write(str(pr))
          outfile.write("\n")
    else:
      for pr in pred[0]:
        print(pr)        

    #test_accuracy = evaluate_accuracy(test_prediction.eval(feed_dict={keep_prob: 1.0}), test_labels)
    #print("test accuracy %g"%test_accuracy)
    
  
  
