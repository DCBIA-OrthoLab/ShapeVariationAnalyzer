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

	num_classes=len(pickle_file)
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

	plt.subplot(221)
	plt.scatter(dataset[:,0],dataset[:,1],alpha=0.5,c=labels,cmap=plt.cm.get_cmap('nipy_spectral',8))
	plt.title('Original data (218 shapes)')
	

	plt.subplot(222)
	plt.scatter(dataset_pca[:,0],dataset_pca[:,1],edgecolor='none',alpha=0.5,c=labels,cmap=plt.cm.get_cmap('nipy_spectral',8))
	plt.title('Original data reconstructed with pca')
	
	plt.subplot(223)
	pca.fit(dataset_res)
	dataset_res_pca=pca.transform(dataset_res)
	dataset_res_new = pca.inverse_transform(dataset_res_pca)
	plt.scatter(dataset_res[:,0],dataset_res[:,1],alpha=0.5,edgecolor='none',c=labels_res,cmap=plt.cm.get_cmap('nipy_spectral',8))
	plt.title('Resampled data (520 shapes)')
	

	plt.subplot(224)
	plt.scatter(dataset_res_pca[:,0],dataset_res_pca[:,1],edgecolor='none',alpha=0.5,c=labels_res,cmap=plt.cm.get_cmap('nipy_spectral',8))
	plt.title('Resampled data reconstructed with pca')

	for i in range(1,4,2):
		plt.subplot(2,2,i+1)
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
	#outputdataPath=args.dirwithSubGenerated
	total_features = 15
	nb_points = 1002
	train_size = 7000
	valid_size = 1000
	test_size = 72

	inputdata = inputData.inputData()
	data_folders = inputdata.get_folder_classes_list(dataPath)
	pickled_datasets = inputdata.maybe_pickle(data_folders, 5)

	dataset,labels = get_labels(pickled_datasets)

	dataset_res,labels_res=generate_with_SMOTE(dataset,labels)
	data_for_training=dataset_res.reshape(len(labels_res),nb_points,total_features)

	print('dataset_res',np.shape(data_for_training))
	print('labels_res',np.shape(labels_res))
	print('labels_res',labels_res)

	PCA_plot(dataset,labels,dataset_res,labels_res)

	force = False
	# #pickled_generated_datasets = inputdata.maybe_pickle(data_folders, 5)
	# dataset_names = []

	# for classfolder in data_folders :
	# 	set_filename = classfolder + 'fortrain.pickle'
	# 	dataset_names.append(set_filename)
	# 	if os.path.exists(set_filename) and not force:
	# 		print('%s already present - Skipping pickling.' % set_filename)
	# 	else:
	# 		vtklist = glob.glob(os.path.join(classfolder, "*.vtk"))
	# 		print('Pickling %s.' % set_filename)
	# 	try:
	# 		with open(set_filename, 'wb') as f:
	# 			pickle.dump(data_for_training, f, pickle.HIGHEST_PROTOCOL)
	# 	except Exception as e:
	# 		print('Unable to save data to', set_filename, ':', e)



	# print(dataset_names)

	# valid_dataset, valid_labels, train_dataset, train_labels = inputdata.merge_datasets(dataset_names, train_size, valid_size=0)

	# print('Training:', train_dataset.shape, train_labels.shape)
	# #print('Validation:', valid_dataset.shape, valid_labels.shape)
	# print('Testing:', test_dataset.shape, test_labels.shape)

	pickle_file = 'datasets.pickle'
	total_number_shapes=data_for_training.shape[0]

	shuffled_dataset, shuffled_labels = inputdata.randomize(data_for_training, labels_res)
	#Divides data vector in 3 groups randomly : training, validation, testing
	print('premier index',int(total_number_shapes))
	try:

		num_train = int(args.train_size*total_number_shapes)
		num_valid = int((total_number_shapes - num_train)*args.validation_size)

		f = open(pickle_file, 'wb')
		
		save = {
        'train_dataset': shuffled_dataset[0:num_train],
        'train_labels': shuffled_labels[0:num_train],
        'valid_dataset': shuffled_dataset[num_train: num_train + num_valid],
        'valid_labels': shuffled_labels[num_train: num_train + num_valid],
        'test_dataset': shuffled_dataset[num_train + num_valid:],
        'test_labels': shuffled_labels[num_train + num_valid:]
		}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    	#f.close()
		
	except Exception as e:
		print('Unable to save data to', pickle_file, ':', e)
		raise




	



    




