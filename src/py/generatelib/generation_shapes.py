#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Principal components analysis (PCA)
=========================================================

These figures aid in illustrating how a point cloud
can be very flat in one direction--which is where PCA
comes in to choose a direction that is not flat.

"""

# Authors: Gael Varoquaux
#          Jaques Grobler
#          Kevin Hughes
# License: BSD 3 clause

from sklearn.decomposition import PCA
import numpy as np
from scipy import stats
import vtk
import os
import argparse
import timeit
import pickle
import random
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pprint 
import inputData
from sklearn.decomposition import PCA
import math
import inputData
import glob


# #############################################################################
# Generate data

parser = argparse.ArgumentParser(description='Shape Variation Analyzer')
#parser.add_argument('--model', type=str, help='pickle file with the pca decomposition', required=True)
#parser.add_argument('--shapeDir', type=str, help='Directory with vtk files .vtk', required=True)
parser.add_argument('--dataPath', action='store', dest='dirwithSub', help='folder with subclasses', required=True)
parser.add_argument('--train_size', help='train ratio', type=float, default=0.8)
parser.add_argument('--validation_size', help='validation ratio from test data', default=0.5, type=float)
#parser.add_argument('-outputdataPath', action='store', dest='dirwithSubGenerated', help='folder with subclasses after generation of data', required=True)
#parser.add_argument('--outputGenerated', help='output folder for shapes', default='./out')
#parser.add_argument('--num_shapes', type=int, help='number shapes to be generated', default=10)
#parser.add_argument('--meanShape',help='mean shape', required=True)


def readData(shapedir):

    #Read data from vtk files
      

    print("loading data ......")
    print("+++++++Read the surface shape data+++++++")    

    vtkdirshapes = os.listdir(shapedir)

    y_design = []
    numpoints = -1
    nshape = 0
    firstshapedata = 0

    for vtkfilename in vtkdirshapes:
        if vtkfilename.endswith((".vtk")):
            print("Reading", vtkfilename)
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(os.path.join(shapedir, vtkfilename))
            reader.Update()
            shapedata = reader.GetOutput()
            shapedatapoints = shapedata.GetPoints()

            if firstshapedata == 0:
                firstshapedata = shapedata

            y_design.append([])

            if numpoints == -1:
                numpoints = shapedatapoints.GetNumberOfPoints()

            if numpoints != shapedatapoints.GetNumberOfPoints():
                print("WARNING! The number of points is not the same for the shape:", vtkfilename)

            for i in range(shapedatapoints.GetNumberOfPoints()):
                p = shapedatapoints.GetPoint(i)
                y_design[nshape].append(p)

            nshape+=1
            
    y_design = np.array(y_design)
    return y_design.reshape(y_design.shape[0], -1), firstshapedata

def writeData(data_for_training,outputdataPath):
#write data in a vtk file


	vtkdirshapes = os.listdir(outputdataPath)
	for vtkfilename in vtkdirshapes:
		if vtkfilename.endswith((".vtk")):
			print("Writing", vtkfilename)
			writer = vtk.vtkPolyDataWriter()
			writer.SetInput(data_for_training)
			writer.SetFileName(os.path.join(outputdataPath),vtkfilename)
			writer.Write()




def get_labels(pickle_file):
#get labels of a dataset and returns the labels array and the dataset with features

	#num_classes=len(pickle_file)
	#num_shapes = 268 #should be changed!!
	labels = []
	shape =[]
	dataset_concatenated =[]

	for label, pickle_file in enumerate(pickle_file):

			try:
				with open(pickle_file,'rb') as f:

					dataset=pickle.load(f)
					shape_dataset = np.shape(dataset)
					num_shapes_per_group = shape_dataset[0]
					l=[label]*num_shapes_per_group
					labels.extend(l)

					dataset_concatenated.extend(dataset)

			except Exception as e:
				print('Unable to process', pickle_file,':',e)
				raise

	features=np.array(dataset_concatenated)
	shape_features=np.shape(features)

	return features.reshape(-1,shape_features[1]*shape_features[2]), np.array(labels)

def generate_data(pca_model):
#generate data thanks to pca decomposition (not used)
	
	print("Generating data ...")

	pca = pca_model["pca"]
	X_ = pca_model["X_"] 
	X_pca_ = pca_model["X_pca_"]
	X_pca_var = pca_model["X_pca_var"] 
	print('Variance',X_pca_var)
	print('Mean',X_pca_)

	#between -1 and 1
	alpha = 2.0*(np.random.random_sample(np.size(X_pca_))) - 1.0
	print('alpha', alpha)

	data_compressed = 1.5*X_pca_var * alpha + X_pca_
	print('data compressed',data_compressed)
	data_generated = pca.inverse_transform(data_compressed) + X_

	return data_generated


def generate_with_SMOTE(dataset,labels):

#generate data thanks to SMOTE algorithm, it balances different groups


	sm=SMOTE(kind='regular')
	print('shape dataset',dataset.shape)
	print('shape labels',labels.shape)
	dataset_res, labels_res = sm.fit_sample(dataset,labels)
	print('shape dataset resampled',np.shape(dataset_res),'shape lables resampled',np.shape(labels_res))

	return dataset_res,labels_res



def PCA_plot(dataset,labels,dataset_res,labels_res):

#plot original dat and data resampled after a PCA decomposition
	
	pca = PCA(n_components=2)
	pca.fit(dataset)
	dataset_pca=pca.transform(dataset)
	print('original shape: ',dataset.shape)
	print('transformed shape:',dataset_pca.shape)
	print('Ratio variance',pca.explained_variance_ratio_)
	plt.scatter(dataset[:,0],dataset[:,1],alpha=0.2)
	#dataset_new = pca.inverse_transform(dataset_pca)
	plt.figure(1)
	plt.subplot(121)
	plt.scatter(dataset_pca[:,0],dataset_pca[:,1],edgecolor='none',alpha=0.5,c=labels,cmap=plt.cm.get_cmap('nipy_spectral',8))
	plt.title('Original data with pca (268 shapes)')
	
	pca.fit(dataset_res)
	dataset_res_pca=pca.transform(dataset_res)
	

	plt.subplot(122)
	plt.scatter(dataset_res_pca[:,0],dataset_res_pca[:,1],edgecolor='none',alpha=0.5,c=labels_res,cmap=plt.cm.get_cmap('nipy_spectral',8))
	plt.title('Resampled data with pca (735 shapes)')

	for i in range(1,3):
		plt.subplot(1,2,i)
		plt.xlabel('component 1')
		plt.ylabel('component 2')
		plt.colorbar()


	plt.figure(2)
	plt.plot(np.cumsum(pca.explained_variance_ratio_))
	plt.xlabel('nb of components')
	plt.ylabel('cumulative explained variance')
	plt.show()


if __name__ == '__main__':

	np.set_printoptions(threshold='nan')

	args = parser.parse_args()
	dataPath=args.dirwithSub
	
	train_size = 7000
	valid_size = 1000
	test_size = 72

	inputdata = inputData.inputData()
	total_features = inputdata.NUM_FEATURES
	nb_points = inputdata.NUM_POINTS

	data_folders = inputdata.get_folder_classes_list(dataPath)
	pickled_datasets = inputdata.maybe_pickle(data_folders, 5)

	dataset,labels = get_labels(pickled_datasets)
	dataset_reshaped = dataset.reshape(-1,nb_points,total_features)

	#shuffle real dataset
	shuffled_real_dataset, shuffled_real_labels = inputdata.randomize(dataset_reshaped, labels)
	#reshape the shuffled dataset for generating new data through SMOTE
	shuffled_real_dataset = shuffled_real_dataset.reshape(shuffled_real_dataset.shape[0],-1)

	smote_dataset_res,smote_labels_res=generate_with_SMOTE(shuffled_real_dataset,shuffled_real_labels)

	#plot real dataset and resampled dataset after SMOTE
	PCA_plot(dataset,labels,smote_dataset_res,smote_labels_res)

	smote_dataset_res=smote_dataset_res.reshape(len(smote_labels_res),nb_points,total_features)


	pickle_file = 'datasets.pickle'
	total_number_shapes=smote_dataset_res.shape[0]
	num_real_shapes = dataset.shape[0]
	#shuffling the SMOTE data not the real
	shuffled_dataset, shuffled_labels = inputdata.randomize(smote_dataset_res[num_real_shapes:], smote_labels_res[num_real_shapes:])

	data = np.concatenate([shuffled_real_dataset.reshape(-1),shuffled_dataset.reshape(-1)], axis=0)
	data = data.reshape(-1,1002,14)
	labels = np.concatenate([shuffled_real_labels, shuffled_labels], axis=0)
	#print('shuffled dataset',np.shape(shuffled_dataset))
	#Divides data vector in 3 groups randomly : training, validation, testing

	try:
		num_train = int(0.2*num_real_shapes)
		#num_train = int(args.train_size*total_number_shapes)
		num_valid = int((total_number_shapes - num_train)*args.validation_size)
		

		f = open(pickle_file, 'wb')
		
		save = {
        #'train_dataset': shuffled_dataset[0:num_train],
        #'train_labels': shuffled_labels[0:num_train],
		#'valid_dataset': shuffled_dataset[num_train: num_train + num_valid],
        #'valid_labels': shuffled_labels[num_train: num_train + num_valid],
        #'test_dataset': shuffled_dataset[num_train + num_valid:],
        #'test_labels': shuffled_labels[num_train + num_valid:]

        #train dataset is 20% of the real dataset (not SMOTE)
		'train_dataset': shuffled_real_dataset[0:num_train],
        'train_labels': shuffled_real_labels[0:num_train],

        #valid and test dataset uses SMOTE data
        'valid_dataset': data[num_train: num_train + num_valid],
        'valid_labels': labels[num_train: num_train + num_valid],
        'test_dataset': data[num_train + num_valid:],
        'test_labels': labels[num_train + num_valid:]
		}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    	#f.close()
		
	except Exception as e:
		print('Unable to save data to', pickle_file, ':', e)
		raise




	



    




