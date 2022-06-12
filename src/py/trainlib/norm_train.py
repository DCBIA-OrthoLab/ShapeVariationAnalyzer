import numpy as np
import tensorflow as tf

import pickle as pickle
import argparse
import norm_nn as nn
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime
from math import sqrt
from matplotlib import pyplot

print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser()

# group = parser.add_mutually_exclusive_group(required=True)


parser.add_argument('--pickle', type=str, help='Pickle file, check the script readImages to generate this file. check generate_polys.py in generatelib', required=True)
parser.add_argument('--out', help='Output directory,the output name will be <out>/model-<num step>', default="./")
parser.add_argument('--learning_rate', help='Learning rate, default=1e-5', type=float, default=1e-5)
parser.add_argument('--decay_rate', help='decay rate, default=0.96', type=float, default=0.96)
parser.add_argument('--decay_steps', help='decay steps, default=10000', type=int, default=10000)
parser.add_argument('--batch_size', help='Batch size for evaluation, default=64', type=int, default=1)
parser.add_argument('--num_epochs', help='Number of iterations, default=10', type=int, default=10)
parser.add_argument('--model', help='Model file computed with metatrader_train.py')
parser.add_argument('--lookback', help='Create sets of series of this size to train the network', type=int, default=1)
parser.add_argument('--lookforward', help='Create sets of series of this size for prediction', type=int, default=1)
parser.add_argument('--ps_device', help='Process device, to store memory', type=str, default="/cpu:0")
parser.add_argument('--w_device', help='Work device, does operations', type=str, default="/cpu:0")


args = parser.parse_args()

pickle_file = args.pickle
outvariablesdirname = args.out
learning_rate = args.learning_rate
decay_rate = args.decay_rate
decay_steps = args.decay_steps
batch_size = args.batch_size
num_epochs = args.num_epochs
model = args.model
lookback = args.lookback
lookforward = args.lookforward
ps_device = args.ps_device
w_device = args.w_device

f = open(pickle_file, 'rb')
data = pickle.load(f)
train_dataset = data["train_dataset"]
train_labels = data["train_labels"]
points_to_cells = data["points_to_cells"]
cells_to_points = data["cells_to_points"]

  
train_dataset_shape = train_dataset.shape
train_labels_shape = train_labels.shape


print('Training set', train_dataset.shape, train_labels.shape)
print('Conversion set', points_to_cells.shape, cells_to_points.shape)
print('learning_rate', learning_rate)
print('decay_rate', decay_rate)
print('decay_steps', decay_steps)
print('batch_size', batch_size)
print('num_epochs', num_epochs)

graph = tf.Graph()

with graph.as_default():

  keep_prob = tf.placeholder(tf.float32)
  
  dataset_x = tf.data.Dataset.from_tensor_slices(train_dataset)

  dataset_x = dataset_x.repeat(num_epochs)
  dataset_x = dataset_x.batch(batch_size)
  iterator_x = dataset_x.make_initializable_iterator()
  next_train_batch_x = iterator_x.get_next()

  dataset_y = tf.data.Dataset.from_tensor_slices(train_labels)

  dataset_y = dataset_y.repeat(num_epochs)
  dataset_y = dataset_y.batch(batch_size)
  iterator_y = dataset_y.make_initializable_iterator()
  next_train_batch_y = iterator_y.get_next()
  
  tf_cells_to_points = tf.placeholder(tf.float32,shape=cells_to_points.shape, name="cells_to_points")
  x = tf.placeholder(tf.float32,shape=(batch_size, train_dataset_shape[1], train_dataset_shape[2], train_dataset_shape[3]), name="x")
  y = tf.placeholder(tf.float32, shape=(batch_size, train_labels_shape[1], train_labels_shape[2]), name="y_")
  is_training = tf.placeholder_with_default(tf.Variable(False, dtype=tf.bool, trainable=False),shape=None)

# calculate the loss from the results of inference and the labels

  # with tf.variable_scope("batch_normalization"):
  #   x_norm = tf.layers.batch_normalization(x, training=is_training)
  # with tf.variable_scope("batch_normalization", reuse=True):
  #   y_norm = tf.layers.batch_normalization(y, training=is_training)

  y_conv = nn.inference_rnn(x, cells_to_points=tf_cells_to_points, batch_size=batch_size, keep_prob=keep_prob, training=is_training, ps_device=ps_device, w_device=w_device)
  loss = nn.loss(y_conv, y)
  # setup the training operations
  train_step = nn.training(loss, learning_rate, decay_steps, decay_rate)

  accuracy_eval = nn.evaluation(y_conv, y)

  tf.summary.scalar(loss.op.name, loss)
  tf.summary.scalar('accuracy', accuracy_eval[0])

  summary_op = tf.summary.merge_all()

  init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

  with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()

    if model is not None:
      print("Restoring model:", model)
      saver.restore(sess, model)
    # specify where to write the log files for import to TensorBoard
    now = datetime.now()
    summary_writer = tf.summary.FileWriter(os.path.join(outvariablesdirname, now.strftime("%Y%m%d-%H%M%S")), sess.graph)

    sess.run([iterator_x.initializer, iterator_y.initializer])

    step = 0
    accuracy_t = 0
    while True:
      try:

        next_batch_x, next_batch_y = sess.run([next_train_batch_x, next_train_batch_y])
        _, loss_value, summary, accuracy = sess.run([train_step, loss, summary_op, accuracy_eval], feed_dict={keep_prob: 0.3, y: next_batch_y, x: next_batch_x, tf_cells_to_points: cells_to_points, is_training: True})
        step += 1

        accuracy_t += accuracy[1]
        
        if step % 100 == 0:
          print('OUTPUT: Step %d: loss = %.6f' % (step, loss_value))
          print('Accuracy:', accuracy)
          # output some data to the log files for tensorboard
          summary_writer.add_summary(summary, step)
          summary_writer.flush()

          # less frequently output checkpoint files.  Used for evaluating the model
        if step % 1000 == 0:
          save_path = saver.save(sess, os.path.join(outvariablesdirname, "model"), global_step=step)
          print('Model saved to:', save_path)

      except tf.errors.OutOfRangeError:
        break

    
    print('Step', step)
    print('Accuracy:', accuracy, accuracy_t/step)
    saver.save(sess, os.path.join(outvariablesdirname, "model"), global_step=step)
    
    predictions = []
    for tx in test_dataset:
      try:

        tx = tx.reshape(1,lookback,train_dataset_shape[2])
        y_predict = sess.run(y_conv, feed_dict={keep_prob: 1, x: tx})
        
        print(y_predict)
        predictions.append(y_predict)

      except tf.errors.OutOfRangeError:
        break

    predictions = np.array(predictions).reshape(-1,5)
    predictions_scaled = 2.0*scaler.inverse_transform(predictions)
    predictions_scaled[0] = raw_dataset[train_len + lookback] + predictions_scaled[0]

    for i in range(1, len(predictions_scaled)):
      predictions_scaled[i] += predictions_scaled[i - 1]

    rmse = sqrt(mean_squared_error(raw_dataset[train_len + lookback:-1], predictions_scaled))
    print('Test RMSE: %.3f' % rmse)

    pyplot.plot(raw_dataset[train_len + lookback:-1,0])
    pyplot.plot(predictions_scaled[:,0])
    pyplot.show()
    
  
  
