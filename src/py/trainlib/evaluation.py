
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

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Model file computed with train.py', required=True)
parser.add_argument('--out', help='Directory to write output of evaluation', required=True, type=str)
parser.add_argument('--pickle', help='Pickle file, check the script readImages to generate this file.', required=True,)
parser.add_argument('--num_labels', help='Number of labels', type=int, default=7)

args = parser.parse_args()

pickle_file = args.pickle
outsummarydirname = args.out
model = args.model
num_labels = args.num_labels
batch_size = 1

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

size_features = test_dataset.shape[1]

print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
print('num_labels', num_labels)

graph = tf.Graph()

with graph.as_default():

# run inference on the input data
  x = tf.placeholder(tf.float32,shape=(None, size_features))
  y_ = tf.placeholder(tf.float32, shape=(None, num_labels))
  
  keep_prob = tf.placeholder(tf.float32)

  y_conv = nn.inference(x, size_features, num_labels=num_labels)
  
  logits_eval = tf.nn.softmax(y_conv)
  label_eval = tf.argmax(logits_eval, axis=1)
  
  auc_eval,fn_eval,fp_eval,tn_eval,tp_eval = nn.metrics(logits_eval, y_)

  tf.summary.scalar("auc_0", auc_eval[0])
  tf.summary.scalar("auc_1", auc_eval[1])
  tf.summary.scalar("fn_eval", fn_eval[1])
  tf.summary.scalar("fp_eval", fp_eval[1])
  tf.summary.scalar("tn_eval", tn_eval[1])
  tf.summary.scalar("tp_eval", tp_eval[1])
  

  summary_op = tf.summary.merge_all()

  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, model)

    now = datetime.now()
    summary_writer = tf.summary.FileWriter(os.path.join(outsummarydirname, now.strftime("%Y%m%d-%H%M%S")), sess.graph)

    totalacc = 0.0
    totalstep = 0
    
    # _, accuracy, auc = sess.run([y_conv, accuracy_eval, auc_eval], feed_dict={x: valid_dataset[0:1], y_: valid_labels[0:1], keep_prob: 1.0})

    # print('Evaluate validation dataset') 
    # print('Step,Accuracy,Auc')
    # for step in range(len(valid_dataset)):
    #   _, accuracy, auc = sess.run([y_conv, accuracy_eval, auc_eval], feed_dict={x: valid_dataset[step:step + 1], y_: valid_labels[step:step + 1], keep_prob: 1.0})
    #   print("%d,%.3f,%.3f"%(step,accuracy[0],auc[0]))

    #   totalacc += accuracy[0]
    #   totalstep += 1

    # print("Valid accuracy %.3f"%(totalacc/totalstep))

    print('Evaluate test dataset')
    for step in range(len(test_dataset)):
      
      batch_data = test_dataset[step:step + 1]
      batch_labels = test_labels[step: step + 1]

      logits,label,auc,summary = sess.run([logits_eval,label_eval,auc_eval,summary_op], feed_dict={x: batch_data, y_: batch_labels, keep_prob: 1})

      summary_writer.add_summary(summary, step)
      summary_writer.flush()

      # print(logits,label)
      # print("%d,%.3f,%.3f"%(step,accuracy[0],auc[0]))

      # totalacc += accuracy[0]
      # totalstep += 1
    
    # print("Test accuracy %.3f"%(totalacc/totalstep))
    
  
  
