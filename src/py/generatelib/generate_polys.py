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
import numpy as np



# #############################################################################
# Generate data

parser = argparse.ArgumentParser(description='Shape Variation Analyzer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--model', type=str, help='pickle file with the pca decomposition', required=True)
#parser.add_argument('--shapeDir', type=str, help='Directory with vtk files .vtk', required=True)
parser.add_argument('--dataPath', action='store', dest='dirwithSub', help='folder with subclasses', required=True)
parser.add_argument('--template', help='Sphere template, computed using SPHARM-PDM', type=str, required=True)
parser.add_argument('--train_size', help='train ratio', type=float, default=0.8)
parser.add_argument('--validation_size', help='validation ratio from test data', default=0.5, type=float)
parser.add_argument('--out', dest="pickle_file", help='Pickle file output', default="datasets.pickle", type=str)


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

def get_conversion_matrices(geometry):
	
	points_to_cells = np.zeros((geometry.GetNumberOfCells(), geometry.GetNumberOfPoints()))
	
	for cid in range(geometry.GetNumberOfCells()):
		pointidlist = vtk.vtkIdList()
		geometry.GetCellPoints(cid, pointidlist)
		for pid in range(pointidlist.GetNumberOfIds()):
			points_to_cells[cid][pointidlist.GetId(pid)] = 1

	cells_to_points = np.zeros((geometry.GetNumberOfPoints(), geometry.GetNumberOfCells()))

	for pid in range(geometry.GetNumberOfPoints()):
		pointidlist = vtk.vtkIdList()
		geometry.GetPointCells(pid, pointidlist)
		for cid in range(pointidlist.GetNumberOfIds()):
			cells_to_points[pid][pointidlist.GetId(cid)] = 1

	return points_to_cells, cells_to_points

def get_normals(vtkclassdict):

	inputdata = inputData.inputData()
	labels = []
	dataset_concatenated = []

	# This looks really confusing but is really not

	for folderclass, vtklist in vtkclassdict.items():

		try:
			with open(folderclass + ".pickle",'rb') as f:

				dataset=pickle.load(f)
				normal_features = []

				for vtkfilename in vtklist:

					#We'll load the same files and get the normals
					features = inputdata.load_features(vtkfilename, feature_points=["Normals"])
					normal_features.append(features)

				
				#This reshaping stuff is to get the list of points, i.e., all connected points
				#and the corresponding label which is the normal in this case
				#The data in the dataset contains lists with different sizes
				normal_features = np.array(normal_features)
				
				featshape = np.shape(normal_features)				
				labels.extend(normal_features.reshape(featshape[0], featshape[1], -1))

				dsshape = np.shape(dataset)
				dataset_concatenated.extend(dataset.reshape(dsshape[0], dsshape[2], dsshape[3], -1))

		except Exception as e:
			print('Unable to process', pickle_file,':',e)
			raise

	# lens = np.array([len(dataset_concatenated[i]) for i in range(len(dataset_concatenated))])
	# mask = np.arange(lens.max()) < lens[:,None]
	# padded = np.zeros(mask.shape + (3,))
	# padded[mask] = np.vstack((dataset_concatenated[:]))

	# return np.array(padded), np.array(labels)
	return np.array(dataset_concatenated), np.array(labels)

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
	
	pca = PCA(n_components=200)
	pca.fit(dataset)
	dataset_pca=pca.transform(dataset)
	print('original shape: ',dataset.shape)
	print('transformed shape:',dataset_pca.shape)
	#print('Ratio variance',pca.explained_variance_ratio_)
	#plt.scatter(dataset[:,0],dataset[:,1],alpha=0.2)
	#dataset_new = pca.inverse_transform(dataset_pca)
	plt.figure(2)
	plt.subplot(121)
	plt.scatter(dataset_pca[:,0],dataset_pca[:,1],edgecolor='none',alpha=0.5,c=labels,cmap=plt.cm.get_cmap('nipy_spectral',np.shape(np.unique(labels))[0]))
	plt.title('Original data with pca (' + str(dataset.shape[0]) + ' samples)')
	
	#pca.fit(dataset_res)
	dataset_res_pca=pca.transform(dataset_res)

	plt.subplot(122)
	plt.scatter(dataset_res_pca[:,0],dataset_res_pca[:,1],edgecolor='none',alpha=0.5,c=labels_res,cmap=plt.cm.get_cmap('nipy_spectral',np.shape(np.unique(labels_res))[0]))
	plt.title('Resampled data with pca (' + str(dataset_res_pca.shape[0]) + ' samples)')

	for i in range(1,3):
		plt.subplot(1,2,i)
		plt.xlabel('component 1')
		plt.ylabel('component 2')
	
		plt.colorbar()

	
	cumsum = np.cumsum(pca.explained_variance_ratio_)
	plt.figure(1)
	plt.plot(cumsum)
	plt.xlabel('nb of components')
	plt.ylabel('cumulative explained variance')
	plt.axhline(y=0.95, linestyle=':', label='.95 explained', color="#f23e3e")
	numcomponents = len(np.where(cumsum < 0.95)[0])
	plt.axvline(x=numcomponents, linestyle=':', label=(str(numcomponents) + ' components'), color="#31f9ad")
	plt.legend(loc=0)
	

	histo = np.bincount(labels)
	histo_range = np.array(range(histo.shape[0]))
	plt.figure(3)
	plt.bar(histo_range, histo)
	plt.xlabel('Groups')
	plt.ylabel('Number of samples')

	for xy in zip(histo_range, histo):
	    plt.annotate(xy[1], xy=xy, ha="center", color="#4286f4")

	plt.show()
	


if __name__ == '__main__':

	np.set_printoptions(threshold='nan')

	args = parser.parse_args()
	dataPath=args.dirwithSub
	pickle_file = args.pickle_file
	template = args.template

	reader = vtk.vtkPolyDataReader()
	reader.SetFileName(template)
	reader.Update()

	points_to_cells, cells_to_points = get_conversion_matrices(reader.GetOutput())

	# Get the data from the folders with vtk files
	inputdata = inputData.inputData()
	data_folders = inputdata.get_folder_classes_list(dataPath)

	pickled_datasets = inputdata.maybe_pickle(data_folders, 5, feature_polys=["Points"])
	# Create the labels, i.e., enumerate the groups

	vtklistdict = inputdata.get_vtklist(data_folders)

	dataset,labels = get_normals(vtklistdict)

	# Comput the total number of shapes and train/test size
	total_number_shapes=dataset.shape[0]
	num_train = int(args.train_size*total_number_shapes)
	num_valid = int((total_number_shapes - num_train)*args.validation_size)

	# Randomize the original dataset
	shuffled_dataset, shuffled_labels = inputdata.randomize(dataset, labels)
	
	dataset_res = shuffled_dataset
	labels_res = shuffled_labels

	# SANITY CHECKS
	print('dataset',np.shape(dataset))
	print('labels',np.shape(labels))
	print('dataset_res',np.shape(dataset_res))
	print('labels_res',np.shape(labels_res))

	print('num_train', num_train)
	print('num_valid', num_valid)
	print('num_test', total_number_shapes - num_valid - num_train)

	print("points_to_cells", np.shape(points_to_cells))
	print("cells_to_points", np.shape(cells_to_points))

	# PCA_plot(dataset,labels,dataset_res,labels_res)

	try:

		f = open(pickle_file, 'wb')
		
		save = {
        'train_dataset': dataset_res,
        'train_labels': labels_res,
		'valid_dataset': dataset[num_train: num_train + num_valid],
        'valid_labels': labels[num_train: num_train + num_valid],
        'test_dataset': dataset[num_train + num_valid:],
        'test_labels': labels[num_train + num_valid:],
        'points_to_cells': points_to_cells, 
        'cells_to_points': cells_to_points        
		}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    	#f.close()
		
	except Exception as e:
		print('Unable to save data to', pickle_file, ':', e)
		raise

