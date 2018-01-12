
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
import vtk
import glob

print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Model file computed with regularization_train.py', required=True)
parser.add_argument('--sampleMesh', help='Evaluate an image sample in vtk format')
parser.add_argument('--sampleDir', help='Evaluate a directory with vtk files')
parser.add_argument('--out', help='Write output of evaluation', default=None, type=str)

args = parser.parse_args()

sampleMesh = args.sampleMesh
sampleDir = args.sampleDir
outfilename = args.out
model = args.model
num_labels = 8
num_features = 3 + num_labels + 4
batch_size = 1

def load_features(shape):

  try:
      print("Reading:", shape)
      reader_poly = vtk.vtkPolyDataReader()
      reader_poly.SetFileName(shape)
      # print "shape : " + shape

      reader_poly.Update()
      geometry = reader_poly.GetOutput()

      # --------------------------------- #
      # ----- GET ARRAY OF FEATURES ----- #
      # --------------------------------- #

      # *****
      # ***** Get normals (3 useful components) - already normalized *****
      normalArray = geometry.GetPointData().GetNormals()
      nbCompNormal = normalArray.GetElementComponentSize() - 1  # -1 car 4eme comp = 1ere du pt suivant

      # ***** Get distances to each mean group (nbGlicroups components) and normalization *****
      listGroupMean = list()
      for i in range(0, num_labels):
          name = "distanceGroup" + str(i)
          temp = geometry.GetPointData().GetScalars(name)
          temp_range = temp.GetRange()
          temp_min, temp_max = temp_range[0], temp_range[1]
          for j in range(0, geometry.GetNumberOfPoints()):
              if temp_max:
                  temp.SetTuple1(j, 2 * (temp.GetTuple1(j) - temp_min) / (temp_max) - 1)
              else:
                  temp.SetTuple1(j, 2 * (temp.GetTuple1(j) - temp_min) / (1) - 1)     # CHANGEEEER
          listGroupMean.append(temp)

      # ***** Get Curvatures and value for normalization (4 components) *****
      meanCurvName = "Mean_Curvature"
      meanCurvArray = geometry.GetPointData().GetScalars(meanCurvName)
      meanCurveRange = meanCurvArray.GetRange()
      meanCurveMin, meanCurveMax = meanCurveRange[0], meanCurveRange[1]
      meanCurveDepth = meanCurveMax - meanCurveMin

      maxCurvName = "Maximum_Curvature"
      maxCurvArray = geometry.GetPointData().GetScalars(maxCurvName)
      maxCurveRange = maxCurvArray.GetRange()
      maxCurveMin, maxCurveMax = maxCurveRange[0], maxCurveRange[1]
      maxCurveDepth = maxCurveMax - maxCurveMin

      minCurvName = "Minimum_Curvature"
      minCurvArray = geometry.GetPointData().GetScalars(minCurvName)
      minCurveRange = minCurvArray.GetRange()
      minCurveMin, minCurveMax = minCurveRange[0], minCurveRange[1]
      minCurveDepth = minCurveMax - minCurveMin

      gaussCurvName = "Gauss_Curvature"
      gaussCurvArray = geometry.GetPointData().GetScalars(gaussCurvName)
      gaussCurveRange = gaussCurvArray.GetRange()
      gaussCurveMin, gaussCurveMax = gaussCurveRange[0], gaussCurveRange[1]
      gaussCurveDepth = gaussCurveMax - gaussCurveMin

      # For each point of the current shape
      currentData = np.ndarray(shape=(geometry.GetNumberOfPoints(), num_features), dtype=np.float32)
      for i in range(0, geometry.GetNumberOfPoints()):

          # Stock normals in currentData
          for numComponent in range(0, nbCompNormal):
              currentData[i, numComponent] = normalArray.GetComponent(i, numComponent)

          for numComponent in range(0, num_labels):
              currentData[i, numComponent + nbCompNormal] = listGroupMean[numComponent].GetTuple1(i)

          value = 2 * (meanCurvArray.GetTuple1(i) - meanCurveMin) / meanCurveDepth - 1
          currentData[i, num_labels + nbCompNormal] = value

          value = 2 * (maxCurvArray.GetTuple1(i) - maxCurveMin) / maxCurveDepth - 1
          currentData[i, num_labels + nbCompNormal + 1] = value

          value = 2 * (minCurvArray.GetTuple1(i) - minCurveMin) / minCurveDepth - 1
          currentData[i, num_labels + nbCompNormal + 2] = value

          value = 2 * (gaussCurvArray.GetTuple1(i) - gaussCurveMin) / gaussCurveDepth - 1
          currentData[i, num_labels + nbCompNormal + 3] = value


  except IOError as e:
      print('Could not read:', shape, ':', e, '- it\'s ok, skipping.')

  # print('Full dataset tensor:', dataset.shape)
  # print('Mean:', np.mean(dataset))
  # print('Standard deviation:', np.std(dataset))
  return currentData

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

if sampleMesh != None:
  valid_dataset = load_features(sampleMesh)
  valid_dataset = valid_dataset.reshape(1, -1)
  batch_size = 1
else:
  vtklist = glob.glob(os.path.join(sampleDir, "*.vtk"))
  valid_dataset = []
  for vtkfile in vtklist:
    print("Reading:", vtkfile)
    valid_dataset.append(load_features(vtkfile))
  valid_dataset = np.array(valid_dataset)
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
    sess.run(tf.initialize_all_variables())
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
    
  
  
