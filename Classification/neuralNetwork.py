import os
import numpy as np
import pandas as pd
# import vtk
import inputData
# from six.moves import cPickle as pickle



class neuralNetwork():
	def __init__(self, parent=None):
		if parent:
			parent.title = " "

		self.NUM_HIDDEN_LAYERS = 2
		import tensorflow as tf
		input_Data = inputData.inputData()


	# ----------------------------------------------------------------------------- #
	#                                 Neural Network
	# ----------------------------------------------------------------------------- #


	## Performance measures of the network
	# Computation of : 	- Accuracy
	# 					- Precision (PPV)
	# 					- Sensitivity (TPR)
	# 					- Confusion matrix
	def accuracy(predictions, labels):
		# Accuracy
		accuracy = (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

		# Confusion matrix
		[nbSamples, column] = predictions.shape
		actu = np.zeros(nbSamples)
		pred = np.zeros(nbSamples)
		for i in range(0, nbSamples):
			actu[i] = np.argmax(labels[i,:])
			pred[i] = np.argmax(predictions[i,:])
		y_actu = pd.Series(actu, name='Actual')
		y_pred = pd.Series(pred, name='Predicted')
		df_confusion = pd.crosstab(y_actu, y_pred)

		# PPV and TPR
		TruePos_sum = int(np.sum(predictions[:, 1] * labels[:, 1]))
		PredPos_sum = int(max(np.sum(predictions[:, 1]), 1)) #Max to avoid to divide by 0
		PredNeg_sum = np.sum(predictions[:, 0])
		RealPos_sum = int(np.sum(labels[:, 1]))

		if not PredPos_sum :
			PPV = 0
		else:
			PPV = 100.0 *TruePos_sum / PredPos_sum # Positive Predictive Value, Precision
		if not RealPos_sum:
			TPR = 0
		else:
			TPR = 100.0 *TruePos_sum / RealPos_sum  # True Positive Rate, Sensitivity

		return accuracy, df_confusion, PPV, TPR


	def weight_variable(shape, name=None):
		"""Create a weight variable with appropriate initialization."""
		# initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name = name)

	def bias_variable(shape, name=None):
		"""Create a bias variable with appropriate initialization."""
		# initial = tf.constant(0.1, shape=shape)
		return tf.Variable(tf.constant(0.1, shape=shape),name = name)

	def bias_weights_creation(nb_hidden_nodes_1=0, nb_hidden_nodes_2=0):
		weightsDict = dict()
		if not NUM_HIDDEN_LAYERS:
			W_fc1 = weight_variable([input_Data.NUM_POINTS * input_Data.NUM_FEATURES, input_Data.NUM_CLASSES], "W_fc1")
			b_fc1 = bias_variable([input_Data.NUM_CLASSES],"b_fc1")

			weightsDict = {"W_fc1": W_fc1, "b_fc1": b_fc1}

		elif NUM_HIDDEN_LAYERS == 1:
			W_fc1 = weight_variable([input_Data.NUM_POINTS * input_Data.NUM_FEATURES, nb_hidden_nodes_1], "W_fc1")
			b_fc1 = bias_variable([nb_hidden_nodes_1],"b_fc1")

			W_fc2 = weight_variable([nb_hidden_nodes_1, input_Data.NUM_CLASSES], "W_fc2")
			b_fc2 = bias_variable([input_Data.NUM_CLASSES],"b_fc2")


			weightsDict = {"W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2}
		elif NUM_HIDDEN_LAYERS == 2:
			W_fc1 = weight_variable([input_Data.NUM_POINTS * input_Data.NUM_FEATURES, nb_hidden_nodes_1], "W_fc1")
			b_fc1 = bias_variable([nb_hidden_nodes_1],"b_fc1")

			W_fc2 = weight_variable([nb_hidden_nodes_1, nb_hidden_nodes_2], "W_fc2")
			b_fc2 = bias_variable([nb_hidden_nodes_2],"b_fc2")

			W_fc3 = weight_variable([nb_hidden_nodes_2, input_Data.NUM_CLASSES], "W_fc3")
			b_fc3 = bias_variable([input_Data.NUM_CLASSES],"b_fc3")

			weightsDict = dict()
			weightsDict = {"W_fc1": W_fc1, "b_fc1": b_fc1, "W_fc2": W_fc2, "b_fc2": b_fc2, "W_fc3": W_fc3, "b_fc3": b_fc3}

		return weightsDict


	# Model.
	def model(data, weightsDict):

		if not NUM_HIDDEN_LAYERS:
			with tf.name_scope('FullyConnected1'):
				h_fc1 = tf.matmul(data, W_fc1) + b_fc1
				valren = h_fc1

		elif NUM_HIDDEN_LAYERS == 1:
			with tf.name_scope('FullyConnected1'):

				h_fc1 = tf.matmul(data, W_fc1) + b_fc1
				h_relu1 = tf.nn.relu(h_fc1)

			with tf.name_scope('FullyConnected2'):

				h_fc2 = tf.matmul(h_relu1, W_fc2) + b_fc2
				valren = h_fc2

		elif NUM_HIDDEN_LAYERS == 2:
			with tf.name_scope('FullyConnected1'):

				h_fc1 = tf.matmul(data, weightsDict['W_fc1']) + weightsDict['b_fc1']
				h_relu1 = tf.nn.relu(h_fc1)
				# h_relu1 = tf.nn.dropout(h_relu1, keep_prob)

			with tf.name_scope('FullyConnected2'):

				h_fc2 = tf.matmul(h_relu1, weightsDict['W_fc2']) + weightsDict['b_fc2']
				h_relu2 = tf.nn.relu(h_fc2)
				# h_relu2 = tf.nn.dropout(h_relu2, keep_prob)

			with tf.name_scope('FullyConnected3'):

				h_fc3 = tf.matmul(h_relu2, weightsDict['W_fc3']) + weightsDict['b_fc3']
				valren = h_fc3

		return valren, weightsDict

	regularization = True
	def loss(logits, tf_train_labels, lambda_reg, weightsDict):
		if not regularization:
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
		else:
			if not NUM_HIDDEN_LAYERS:
				norms = tf.nn.l2_loss(weightsDict['W_fc1']) 
			elif NUM_HIDDEN_LAYERS == 1:
				norms = tf.nn.l2_loss(weightsDict['W_fc1']) + tf.nn.l2_loss(weightsDict['W_fc2'])
			elif NUM_HIDDEN_LAYERS == 2:
				norms = tf.nn.l2_loss(weightsDict['W_fc1']) + tf.nn.l2_loss(weightsDict['W_fc2']) + tf.nn.l2_loss(weightsDict['W_fc3'])

			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + lambda_reg*norms)
		return loss

