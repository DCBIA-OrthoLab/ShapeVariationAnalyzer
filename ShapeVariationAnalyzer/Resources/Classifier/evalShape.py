import os
import sys
from six.moves import cPickle as pickle
import neuralNetwork as nn
import numpy as np
import tensorflow as tf

import argparse
import zipfile
import shutil
import json


def reformat_data(dataset, classifier):
    """ Reformat into a shape that's more adapted to the models we're going to train:
        - data as a flat matrix
        - labels as float 1-hot encodings
    """
    dataset = dataset.reshape((-1, classifier.NUM_POINTS * classifier.NUM_FEATURES)).astype(np.float32)
    return dataset


def get_input_shape(data, classifier):
    """ Get features in a matrix (NUM_FEATURES x NUM_POINTS)
    """
    data = data.reshape((-1, classifier.NUM_POINTS * classifier.NUM_FEATURES)).astype(np.float32)
    data = reformat_data(data, classifier)
    return data

# ----------------- #
# ---- RESULTS ---- #
# ----------------- #
def get_result(prediction):
    return np.argmax(prediction[0,:])

def exportModelNetwork(zipPath, outputPath):
    shutil.make_archive(base_name = outputPath, format = 'zip', root_dir = os.path.dirname(zipPath), base_dir = os.path.basename(zipPath))
    return


def main(_):
    # Get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputZip', action='store', dest='inputZip', help='Input zip file which contains the datasets & the parameters for the classifier')
    parser.add_argument('--outputZip', action='store', dest='outputZip', help='Input zip file which the network trained and the results of the classification')

    # parser.add_argument('-inputFile', action='store', dest='inputFile', help='Input file to classify', default = "")

    args = parser.parse_args()

    inputZip = args.inputZip
    outputZip = args.outputZip
    # inputFile = args.inputFile

    basedir = os.path.dirname(inputZip)
    nameDir = os.path.splitext(os.path.basename(inputZip))[0]
    networkDir = os.path.join(basedir, nameDir)

    ouputbaseDir = os.path.dirname(outputZip)
    outputName = os.path.splitext(os.path.basename(outputZip))[0]
    outputPath = os.path.join(ouputbaseDir, outputName)

    if os.path.isdir(networkDir):
        shutil.rmtree(networkDir)
    os.mkdir(networkDir) 

    # Unpack archive
    with zipfile.ZipFile(inputZip) as zf:
        zf.extractall(networkDir)
        zf.extractall(basedir)

    jsonFile = os.path.join(networkDir, 'classifierInfo.json')
    saveModelPath = os.path.join(networkDir, 'CondylesClassifier')
    pickleToClassify = os.path.join(networkDir, 'toClassify.pickle')
    #
    # Create a network for the classification
    #
    if sys.version_info[0] == 3: 
        with open(jsonFile, encoding='utf-8') as f:    
            jsonDict = json.load(f)
    else:
        with open(jsonFile) as f:    
            jsonDict = json.load(f)


    # In case our JSON file doesnt contain a valid Classifier
    if not 'CondylesClassifier' in jsonDict:
        print("Error: Couldn't parameterize the network.")
        print("There is no 'CondylesClassifier' model.")
        return 0


    # If we have the Classifier, set all parameters for the network
    classifier = nn.neuralNetwork()

    # Essential parameters
    if 'NUM_CLASSES' in jsonDict['CondylesClassifier']:
        classifier.NUM_CLASSES = jsonDict['CondylesClassifier']['NUM_CLASSES'] 
    else:
        print("Missing NUM_CLASSES")
    
    if 'NUM_POINTS' in jsonDict['CondylesClassifier']:
        classifier.NUM_POINTS = jsonDict['CondylesClassifier']['NUM_POINTS']
    else:
        print("Missing NUM_POINTS")

    if 'NUM_FEATURES' in jsonDict['CondylesClassifier']:
        classifier.NUM_FEATURES = jsonDict['CondylesClassifier']['NUM_FEATURES']
    else:
        print("Missing NUM_FEATURES")


    if sys.version_info[0] == 2: 
        dictToClassify = pickle.load( open( pickleToClassify, "rb" ))
    else:
        dictToClassify = pickle.load( open( pickleToClassify, "rb" ), encoding='latin1' )
    dictClassified = dict()

    for file in dictToClassify.keys():
        # print(file)
        # Create session, and import existing graph
        # print(shape)
        myData = get_input_shape(dictToClassify[file], classifier)
        session = tf.InteractiveSession()


        new_saver = tf.train.import_meta_graph(saveModelPath + '.meta')
        new_saver.restore(session, saveModelPath)
        graph = tf.Graph().as_default()

        # Get useful tensor in the graph
        tf_data = session.graph.get_tensor_by_name("Inputs_management/input:0")
        data_pred = session.graph.get_tensor_by_name("Predictions/output:0")

        feed_dict = {tf_data: myData}
        data_pred = session.run(data_pred, feed_dict=feed_dict)

        result = get_result(data_pred)
        dictClassified[file] = int(result)
        
    # Save into a JSON file
    with open(os.path.join(networkDir,'results.json'), 'w') as f:
        json.dump(dictClassified, f, ensure_ascii=False, indent = 4)

    # Zip all those files together
    zipPath = networkDir
    exportModelNetwork(zipPath, outputPath)

    return True


if __name__ == '__main__':
    tf.app.run()



