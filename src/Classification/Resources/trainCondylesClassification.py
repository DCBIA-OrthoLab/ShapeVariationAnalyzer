import os
import sys
from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf
import neuralNetwork as nn
# import inputData
import argparse
import json
import shutil
import zipfile

# ----------------------------------------------------------------------------- #
#																				#
# 								Useful functions 								#
#																				#
# ----------------------------------------------------------------------------- #

## Reformat into a shape that's more adapted to the models we're going to train:
#   - data as a flat matrix
#   - labels as float 1-hot encodings
def reformat(dataset, labels, classifier):
    dataset = dataset.reshape((-1, classifier.NUM_POINTS * classifier.NUM_FEATURES)).astype(np.float32)
    labels = (np.arange(classifier.NUM_CLASSES) == labels[:, None]).astype(np.float32)
    return dataset, labels
    
def get_inputs(pickle_file, classifier):

    # Reoad the data generated in pickleData.py
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)

        train_dataset, train_labels = reformat(train_dataset, train_labels, classifier)
        valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, classifier)
        test_dataset, test_labels = reformat(test_dataset, test_labels, classifier)
        print("\nTraining set", train_dataset.shape, train_labels.shape)
        print("Validation set", valid_dataset.shape, valid_labels.shape)
        print("Test set", test_dataset.shape, test_labels.shape)
        print ""

        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


def run_training(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, saveModelPath, classifier):

    #       >>>>>       A RENDRE GENERIQUE !!!!!!!
    if classifier.NUM_HIDDEN_LAYERS == 1:
        nb_hidden_nodes_1 = 2048
        nb_hidden_nodes_2 = 0
    elif classifier.NUM_HIDDEN_LAYERS == 2:
        nb_hidden_nodes_1, nb_hidden_nodes_2 = 2048, 2048

    # Construct the graph
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        with tf.name_scope('Inputs_management'):
            # tf_train_dataset, tf_train_labels = placeholder_inputs(classifier.batch_size, name='data')
            tf_train_dataset = tf.placeholder(tf.float32, shape=(classifier.batch_size, classifier.NUM_POINTS * classifier.NUM_FEATURES), name='tf_train_dataset')
            tf_train_labels = tf.placeholder(tf.int32, shape=(classifier.batch_size, classifier.NUM_CLASSES), name='tf_train_labels')

            keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            tf_valid_dataset = tf.constant(valid_dataset, name="tf_valid_dataset")
            tf_test_dataset = tf.constant(test_dataset)

            tf_data = tf.placeholder(tf.float32, shape=(1,classifier.NUM_POINTS * classifier.NUM_FEATURES), name="input")

        with tf.name_scope('Bias_and_weights_management'):
            weightsDict = classifier.bias_weights_creation(nb_hidden_nodes_1, nb_hidden_nodes_2)    
        
        # Training computation.
        with tf.name_scope('Training_computations'):
            logits, weightsDict = classifier.model(tf_train_dataset, weightsDict)
            
        with tf.name_scope('Loss_computation'):
            loss = classifier.loss(logits, tf_train_labels, classifier.lambda_reg, weightsDict)
        
        
        with tf.name_scope('Optimization'):
            # Optimizer.
            optimizer = tf.train.GradientDescentOptimizer(classifier.learning_rate).minimize(loss)
            # optimizer = tf.train.AdagradOptimizer(classifier.learning_rate).minimize(loss)
        
        # tf.tensor_summary("W_fc1", weightsDict['W_fc1'])
        tf.summary.scalar("Loss", loss)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(weightsDict)

            
        with tf.name_scope('Predictions'):
            # Predictions for the training, validation, and test data.
            train_prediction = tf.nn.softmax(logits)
            valid_prediction = tf.nn.softmax(classifier.model(tf_valid_dataset, weightsDict)[0], name="valid_prediction")

            data_pred = tf.nn.softmax(classifier.model(tf_data, weightsDict)[0], name="output")
            test_prediction = tf.nn.softmax(classifier.model(tf_test_dataset, weightsDict)[0])


        # -------------------------- #
        #       Let's run it         #
        # -------------------------- #
        # 
        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print("Initialized")

            # create log writer object
            writer = tf.summary.FileWriter('./train', graph=graph)

            for epoch in range(0, classifier.num_epochs):
                for step in range(classifier.num_steps):
                    # Pick an offset within the training data, which has been randomized.
                    # Note: we could use better randomization across epochs.
                    offset = (step * classifier.batch_size) % (train_labels.shape[0] - classifier.batch_size)
                    # Generate a minibatch.
                    batch_data = train_dataset[offset:(offset + classifier.batch_size), :]
                    batch_labels = train_labels[offset:(offset + classifier.batch_size), :]
                    # Prepare a dictionary telling the session where to feed the minibatch.
                    # The key of the dictionary is the placeholder node of the graph to be fed,
                    # and the value is the numpy array to feed to it.
                    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:0.7}
                    _, l, predictions, summary = session.run([optimizer, loss, train_prediction, summary_op], feed_dict=feed_dict)
                    # _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)


                    # write log
                    batch_count = 20
                    writer.add_summary(summary, epoch * batch_count + step)


                    if (step % 500 == 0):
                        print("Minibatch loss at step %d: %f" % (step, l))
                        print("Minibatch accuracy: %.1f%%" % classifier.accuracy(predictions, batch_labels)[0])
                        print("Validation accuracy: %.1f%%" % classifier.accuracy(valid_prediction.eval(feed_dict = {keep_prob:1.0}), valid_labels)[0])

            finalaccuracy, mat_confusion, PPV, TPR = classifier.accuracy(test_prediction.eval(feed_dict={keep_prob:1.0}), test_labels)
            print("Test accuracy: %.1f%%" % finalaccuracy)
            print("\n\nConfusion matrix :\n" + str(mat_confusion))
            # print "\n PPV : " + str(PPV)
            # print "\n TPR : " + str(TPR)

            save_path = saver.save(session, saveModelPath, write_meta_graph=True)
            print("Model saved in file: %s" % save_path)
    
    return finalaccuracy

def exportModelNetwork(zipPath):

	# Zipper tout ca :: base_name = la ou on veut zipper+zipname
	shutil.make_archive(base_name = zipPath, format = 'zip', root_dir = os.path.dirname(zipPath), base_dir = os.path.basename(zipPath))

	return



# ----------------------------------------------------------------------------- #
# 																				#
# 						   Passons aux choses serieuses							#
# 																				#
# ----------------------------------------------------------------------------- #
# 
def main(_):
    print "\nTensorFlow current version : " + str(tf.__version__) + "\n"
      
    # Get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-inputZip', action='store', dest='inputZip', help='Input zip file which contains the datasets & the parameters for the classifier', 
                        default = "")

    args = parser.parse_args()

    inputZip = args.inputZip
    basedir = os.path.dirname(inputZip)
    nameDir = os.path.splitext(os.path.basename(inputZip))[0]

    networkDir = os.path.join(basedir, nameDir)
    print "networkDir : " + networkDir

    if os.path.isdir(networkDir):
        shutil.rmtree(networkDir)
    os.mkdir(networkDir) 

    # Unpack archive
    with zipfile.ZipFile(inputZip) as zf:
        zf.extractall(basedir)

    jsonFile = os.path.join(networkDir, 'classifierInfo.json')
    saveModelPath = os.path.join(networkDir, 'CondylesClassifier')
    pickle_file = os.path.join(networkDir, 'datasets.pickle')

    #
    # Create a network for the classification
    #
    with open(jsonFile) as f:    
        jsonDict = json.load(f)


    # In case our JSON file doesnt contain a valid Classifier
    if not jsonDict.has_key('CondylesClassifier'):
        print "Error: Couldn't parameterize the network."
        print "There is no 'CondylesClassifier' model."
        return 0

    # If we have the Classifier, set all parameters for the network
    classifier = nn.neuralNetwork()

    # Essential parameters
    if 'NUM_CLASSES' in jsonDict['CondylesClassifier']:
        classifier.NUM_CLASSES = jsonDict['CondylesClassifier']['NUM_CLASSES'] 
    else:
        print "Missing NUM_CLASSES"
        accuracy = -1

    if 'NUM_POINTS' in jsonDict['CondylesClassifier']:
        classifier.NUM_POINTS = jsonDict['CondylesClassifier']['NUM_POINTS']
    else:
        print "Missing NUM_POINTS"
        accuracy = -1

    if 'NUM_FEATURES' in jsonDict['CondylesClassifier']:
        classifier.NUM_FEATURES = jsonDict['CondylesClassifier']['NUM_FEATURES']
    else:
        print "Missing NUM_FEATURES"
        accuracy = -1

    # TODO: Manage case with incomplete parameterization of the classifier network


    # Specific parameters
    if 'learning_rate' in jsonDict['CondylesClassifier']:
        classifier.learning_rate = jsonDict['CondylesClassifier']['learning_rate']
    else: 
        classifier.learning_rate = 0.0005

    if 'lambda_reg' in jsonDict['CondylesClassifier']:
        classifier.lambda_reg = jsonDict['CondylesClassifier']['lambda_reg']
    else:
        classifier.lambda_reg = 0.01

    if 'num_epochs' in jsonDict['CondylesClassifier']:
        classifier.num_epochs = jsonDict['CondylesClassifier']['num_epochs']
    else:
        classifier.num_epochs = 2

    if 'num_steps'	in jsonDict['CondylesClassifier']:
        classifier.num_steps = jsonDict['CondylesClassifier']['num_steps']
    else:
        classifier.num_steps =  11

    if 'batch_size' in jsonDict['CondylesClassifier']:
        classifier.batch_size = jsonDict['CondylesClassifier']['batch_size']
    else:
        classifier.batch_size = 10

    if 'NUM_HIDDEN_LAYERS' in jsonDict['CondylesClassifier']:
        classifier.NUM_HIDDEN_LAYERS = jsonDict['CondylesClassifier']['NUM_HIDDEN_LAYERS']
    else:
        classifier.NUM_HIDDEN_LAYERS = 2


    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_inputs(pickle_file, classifier)

    accuracy = run_training(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, saveModelPath, classifier)
    jsonDict['CondylesClassifier']['accuracy'] = accuracy
    with open(os.path.join(networkDir,'classifierInfo.json'), 'w') as f:
        json.dump(jsonDict, f, ensure_ascii=False, indent = 4)

    # Zip all those files together
    zipPath = networkDir
    exportModelNetwork(zipPath)

    return 


if __name__ == '__main__':
	tf.app.run()







