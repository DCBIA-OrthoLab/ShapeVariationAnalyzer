
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
from datetime import datetime

print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser(description='Shape Variation Analyzer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--pickle', help='Pickle file, check the script pickleData to generate this file.', required=True)
parser.add_argument('--out', help='Output dirname, default=./out', default="./out")
parser.add_argument('--model', help='Output modelname, default=model, the output name will be <outdir>/model-<num step>', default="model")
parser.add_argument('--learning_rate', help='Learning rate, default=1e-5', type=float, default=1e-5)
parser.add_argument('--decay_rate', help='decay rate, default=0.96', type=float, default=0.96)
parser.add_argument('--decay_steps', help='decay steps, default=10000', type=int, default=10000)
parser.add_argument('--batch_size', help='Batch size for evaluation, default=32', type=int, default=32)
parser.add_argument('--reg_constant', help='Regularization constant, default=0.0', type=float, default=0.0)
parser.add_argument('--num_epochs', help='Number of epochs', type=int, default=10)
parser.add_argument('--num_labels', help='Number of labels', type=int, default=7)

args = parser.parse_args()

pickle_file = args.pickle
outvariablesdirname = args.out
modelname = args.model
learning_rate = args.learning_rate
decay_rate = args.decay_rate
decay_steps = args.decay_steps
batch_size = args.batch_size
num_epochs = args.num_epochs
reg_constant = args.reg_constant
num_labels = args.num_labels

f = open(pickle_file, 'rb')
data = pickle.load(f)
train_dataset = data["train_dataset"]
train_labels = data["train_labels"]
valid_dataset = data["valid_dataset"]
valid_labels = data["valid_labels"]
# test_dataset = data["test_dataset"]
# test_labels = data["test_labels"]
# img_head = data["img_head"]
# img_size = img_head["sizes"]
# img_size = [img_size[3], img_size[2], img_size[1], img_size[0]]


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

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
# test_dataset, test_labels = reformat(test_dataset, test_labels)

size_features = train_dataset.shape[1]


print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
# print('Test set', test_dataset.shape, test_labels.shape)
print('learning_rate', learning_rate)
print('decay_rate', decay_rate)
print('decay_steps', decay_steps)
print('batch_size', batch_size)
print('num_epochs', num_epochs)

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
  tf_train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
  tf_train_labels = tf.data.Dataset.from_tensor_slices(train_labels)

  dataset = tf.data.Dataset.zip((tf_train_dataset, tf_train_labels))
  dataset = dataset.repeat(args.num_epochs)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_initializable_iterator()
  next_train_data, next_train_labels = iterator.get_next()  

  x = tf.placeholder(tf.float32,shape=(None, size_features))
  y_ = tf.placeholder(tf.float32, shape=(None, num_labels))
  
  keep_prob = tf.placeholder(tf.float32)

  #tf_valid_dataset = tf.constant(valid_dataset)
  # tf_test_dataset = tf.constant(test_dataset)

  y_conv = nn.inference(x, size_features, num_labels=num_labels, keep_prob=keep_prob, batch_size=batch_size, regularization_constant=reg_constant, is_training=True)

# calculate the loss from the results of inference and the labels
  loss = nn.loss(y_conv, y_)

  tf.summary.scalar(loss.op.name, loss)

  #intersection_sum, label_sum, example_sum, precision = nn.evaluation(y_conv, y_)

  #tf.summary.scalar ("Precision op", precision)

# setup the training operations
  #train_step = nn.training(loss, learning_rate, decay_steps, decay_rate)
  # setup the summary ops to use TensorBoard

  # setup the training operations
  #train_step = nn.training(loss, learning_rate, decay_steps, decay_rate)
  train_step = nn.training(loss, learning_rate, decay_steps, decay_rate)

  accuracy_eval = nn.evaluation(y_conv, y_)
  tf.summary.scalar(accuracy_eval.op.name, accuracy_eval)

  summary_op = tf.summary.merge_all()

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
    # specify where to write the log files for import to TensorBoard
    now = datetime.now()
    summary_writer = tf.summary.FileWriter(os.path.join(outvariablesdirname, now.strftime("%Y%m%d-%H%M%S")), sess.graph)

    sess.run([iterator.initializer])
    step = 0

    while True:
      try:

        batch_data, batch_labels = sess.run([next_train_data, next_train_labels])

        _, loss_value, summary, accuracy = sess.run([train_step, loss, summary_op, accuracy_eval], feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})

        if step % 100 == 0:
          print('OUTPUT: Step %d: loss = %.3f' % (step, loss_value))
          print('Accuracy = %.3f ' % (accuracy))
          # output some data to the log files for tensorboard
          summary_writer.add_summary(summary, step)
          summary_writer.flush()

          # less frequently output checkpoint files.  Used for evaluating the model
        if step % 1000 == 0:
          save_path = saver.save(sess, os.path.join(outvariablesdirname, modelname), global_step=step)

        step += 1

      except tf.errors.OutOfRangeError:
        break
      
    for i in range(0, np.shape(valid_dataset)[0], batch_size):
      _, accuracy = sess.run([y_conv, accuracy_eval], feed_dict={x: valid_dataset[i:i + batch_size], y_: valid_labels[i:i + batch_size], keep_prob: 1})
      print('Validation accuracy = %.3f ' % (accuracy))

    outmodelname = os.path.join(outvariablesdirname, modelname)
    print('Step:', step)
    print('Saving model:', outmodelname)
    saver.save(sess, outmodelname, global_step=step)

    #test_accuracy = evaluate_accuracy(test_prediction.eval(feed_dict={keep_prob: 1.0}), test_labels)
    #print("test accuracy %g"%test_accuracy)
    
  
  
