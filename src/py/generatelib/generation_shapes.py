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
import json
import csv
import tensorflow as tf



# #############################################################################
# Generate data
#python3 generation_shapes.py --dataPath /work/lpzmateo/data/DL_shapes/shapes --out /work/lpzmateo/data/DL_shapes/DATASETTEST/dataset.pyc
parser = argparse.ArgumentParser(description='Shape Variation Analyzer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataDescription', action='store', dest='csvPath', help='csv file describing each shape', required=True)
# parser.add_argument('--dataPath', action='store', dest='dirwithSub', help='folder with subclasses', required=True)
#parser.add_argument('--template', help='Template sphere, output from SPHARM-PDM or similar tool', required=True)
parser.add_argument('--levels', help='Linear subdivision levels', nargs="+", type=int, default=[8,6,4,2])
parser.add_argument('--train_size', help='train ratio', type=float, default=0.8)
parser.add_argument('--validation_size', help='validation ratio from test data', default=0.5, type=float)
parser.add_argument('--feature_points', help='Extract the following features from the polydatas GetPointData', nargs='+', default=["Normals", "Mean_Curvature", "distanceGroup"], type=str)
parser.add_argument('--feature_cells', help='Extract the following features from the polydatas GetCellData', nargs='+', default=["Points","random", "RANDOM"], type=str)
parser.add_argument('--out', dest="dataset_path", help='Dataset directory output', default="./", type=str)


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
	
	labels=labels.astype(np.int)
	histo = np.bincount(labels)
	histo_range = np.array(range(histo.shape[0]))
	plt.figure(3)
	plt.bar(histo_range, histo)
	plt.xlabel('Groups')
	plt.ylabel('Number of samples')
	plt.title('Before SMOTE')

	for xy in zip(histo_range, histo):
	    plt.annotate(xy[1], xy=xy, ha="center", color="#4286f4")

	labels_res=labels_res.astype(np.int)
	histo = np.bincount(labels_res)
	histo_range = np.array(range(histo.shape[0]))
	plt.figure(4)
	plt.bar(histo_range, histo)
	plt.xlabel('Groups')
	plt.ylabel('Number of samples')
	plt.title('After SMOTE')

	for xy in zip(histo_range, histo):
	    plt.annotate(xy[1], xy=xy, ha="center", color="#4286f4")

	plt.show()
	


def extractDataFromTFRecord(tfr_path):
	record_iterator = tf.python_io.tf_record_iterator(path=tfr_path)
	input_data=[]
	target_data=[]
	for string_record in record_iterator:
	    
	    example = tf.train.Example()
	    example.ParseFromString(string_record)
	    
	    input_data = example.features.feature['input'].float_list.value
	    for i in range(len(input_data)):
	    	input_data[i]=float(input_data[i])

	    
	    target_data = example.features.feature['output'].float_list.value
	    for i in range(len(target_data)):
	    	target_data[i]=float(target_data[i])

	    return np.array(input_data), np.array(target_data)
	    


def force_symlink(file1, file2):
    try:
        os.symlink(file1, file2)
    except FileExistsError:
        os.remove(file2)
        os.symlink(file1, file2)

if __name__ == '__main__':

	#np.set_printoptions(threshold='nan')

	args = parser.parse_args()
	csvPath=args.csvPath
	#pickle_file = args.pickle_file
	dataset_path = args.dataset_path

	records_path=os.path.join(dataset_path,'true_data_tfrecords')

	# Get the data from the folders with vtk files
	inputdata = inputData.inputData()


	#dataset_description_path=inputdata.TFRecord_dataset(csvPath,records_path,min_num_shapes_per_class=5, feature_points=args.feature_points,feature_polys=args.feature_cells)
	dataset_description_path=inputdata.TFRecord_dataset('/work/lpzmateo/source/tests/generatif_test.csv',records_path,5, feature_points=args.feature_points,feature_polys=args.feature_cells, feature_points_output=args.feature_points,feature_polys_output=args.feature_cells)

	print('Generating a new dataset')
	#reading dataset description
	with open(dataset_description_path) as json_data:
		dataset_description = json.load(json_data)




	#Generating the final dataset
	if dataset_description['dataset_type']=='classification':
		print('Classification dataset:')
		#get tfrecord paths
		tfr_paths=dict()
		with open(dataset_description['files_description'], mode='r') as csv_file:
			csv_reader = csv.DictReader(csv_file)

			for row in csv_reader:
				if row['Group']in tfr_paths:
					tfr_paths[row['Group']].append(row['TFRecord Files'])
				else:
					tfr_paths[row['Group']]=[row['TFRecord Files']]

		#Extract Data for each group
		new_dataset=dict()
		new_dataset['train']=[]
		new_dataset['test']=[]
		new_dataset['validation']=[]

		for group , path_list in tfr_paths.items():

			#randomize files
			np_tfr_paths=np.array(path_list)
			permutation=np.random.permutation(np_tfr_paths.shape[0])
			path_list=np_tfr_paths[permutation]

			num_train = int(args.train_size*len(path_list))
			num_valid = int((len(path_list) - num_train)*args.validation_size)

			if (num_train < 6):
				n=6-num_train
				num_train=num_train+n
				num_valid=num_valid-n
				if num_train>len(path_list):
					print('ERROR: not enought examples in the group',group)

			new_dataset['train'].extend(path_list[0:num_train])
			new_dataset['validation'].extend(path_list[num_train:num_train+num_valid])
			new_dataset['test'].extend(path_list[num_train+num_valid:])
		
		print('train examples: ',len(new_dataset['train']))
		print('test examples: ',len(new_dataset['test']))
		print('validation examples: ',len(new_dataset['validation']))

		print(' ')
		print('Generating data using SMOTE ...')

		#Smote on the train set
		input_dataset=[]
		target_dataset=[]
		for file_path in new_dataset['train']:
			input_data,target_data=extractDataFromTFRecord(file_path)
			input_dataset.append(input_data)
			target_dataset.append(target_data[0])

		input_dataset=np.array(input_dataset)
		target_dataset=np.array(target_dataset)

		dataset_res,labels_res=generate_with_SMOTE(input_dataset,target_dataset)
		PCA_plot(input_dataset,target_dataset,dataset_res,labels_res)
		dataset_res=dataset_res[input_dataset.shape[0]:,:]
		labels_res=labels_res[target_dataset.shape[0]:]
		print('Generation done')


		print('Saving SMOTE TFRecord...')
		smote_dir=os.path.join(dataset_path,'smote_data_tfrecords')
		smote_records_paths=inputdata.writeRecords(smote_dir,dataset_res,labels_res,start_id=0,file_name_prefix='TFR_SMOTE_')

		print('Creating soft links...')

		#TRAIN
		train_dir=os.path.join(dataset_path,'train')
		try:
		    os.mkdir(train_dir)
		except:
		    pass

		i = 0

		for file_path in new_dataset['train']:
			force_symlink(file_path,os.path.join(train_dir,'train_'+str(i)+'.tfrecord'))
			i=i+1

		for file_path in smote_records_paths:
			force_symlink(file_path,os.path.join(train_dir,'train_'+str(i)+'.tfrecord'))
			i=i+1


		#TEST
		test_dir=train_dir=os.path.join(dataset_path,'test')
		try:
		    os.mkdir(test_dir)
		except:
		    pass

		i = 0

		for file_path in new_dataset['test']:
			force_symlink(file_path,os.path.join(test_dir,'test_'+str(i)+'.tfrecord'))
			i=i+1

		#VALIDATION
		val_dir=train_dir=os.path.join(dataset_path,'validation')
		try:
		    os.mkdir(val_dir)
		except:
		    pass

		i = 0

		for file_path in new_dataset['validation']:
			force_symlink(file_path,os.path.join(val_dir,'validation_'+str(i)+'.tfrecord'))
			i=i+1

		print('Generating dataset description...')


		del dataset_description['examples_per_class']
		dataset_description['train_examples']=len(new_dataset['train'])+len(smote_records_paths)
		dataset_description['test_examples']=len(new_dataset['test'])
		dataset_description['validation_examples']=len(new_dataset['validation'])


		data_set_info_path=os.path.join(dataset_path,'dataset_description.json')
		try:
			with open(data_set_info_path, 'w') as f:
			    json.dump(dataset_description, f,indent = 4)
		except Exception as e:
			print('Unable to save extraction description to', data_set_info_path, ':', e)

	if dataset_description['dataset_type']=='generation':
		print('Generation dataset:')
		#get tfrecord paths
		tfr_paths=[]
		with open(dataset_description['files_description'], mode='r') as csv_file:
			csv_reader = csv.DictReader(csv_file)

			for row in csv_reader:
				tfr_paths.append(row['TFRecord Files'])


		#Extract Data for each group
		new_dataset=dict()
		new_dataset['train']=[]
		new_dataset['test']=[]
		new_dataset['validation']=[]

		#randomize files
		np_tfr_paths=np.array(tfr_paths)
		permutation=np.random.permutation(np_tfr_paths.shape[0])
		path_list=np_tfr_paths[permutation]

		num_train = int(args.train_size*len(path_list))
		num_valid = int((len(path_list) - num_train)*args.validation_size)

		new_dataset['train'].extend(path_list[0:num_train])
		new_dataset['validation'].extend(path_list[num_train:num_train+num_valid])
		new_dataset['test'].extend(path_list[num_train+num_valid:])
	
		print('train examples: ',len(new_dataset['train']))
		print('test examples: ',len(new_dataset['test']))
		print('validation examples: ',len(new_dataset['validation']))


		print('Creating soft links...')

		#TRAIN
		train_dir=os.path.join(dataset_path,'train')
		try:
		    os.mkdir(train_dir)
		except:
		    pass

		i = 0

		for file_path in new_dataset['train']:
			force_symlink(file_path,os.path.join(train_dir,'train_'+str(i)+'.tfrecord'))
			i=i+1


		#TEST
		test_dir=train_dir=os.path.join(dataset_path,'test')
		try:
		    os.mkdir(test_dir)
		except:
		    pass

		i = 0

		for file_path in new_dataset['test']:
			force_symlink(file_path,os.path.join(test_dir,'test_'+str(i)+'.tfrecord'))
			i=i+1

		#VALIDATION
		val_dir=train_dir=os.path.join(dataset_path,'validation')
		try:
		    os.mkdir(val_dir)
		except:
		    pass

		i = 0

		for file_path in new_dataset['validation']:
			force_symlink(file_path,os.path.join(val_dir,'validation_'+str(i)+'.tfrecord'))
			i=i+1

		print('Generating dataset description...')


		dataset_description['train_examples']=len(new_dataset['train'])
		dataset_description['test_examples']=len(new_dataset['test'])
		dataset_description['validation_examples']=len(new_dataset['validation'])


		data_set_info_path=os.path.join(dataset_path,'dataset_description.json')
		try:
			with open(data_set_info_path, 'w') as f:
			    json.dump(dataset_description, f,indent = 4)
		except Exception as e:
			print('Unable to save extraction description to', data_set_info_path, ':', e)

	#checking data 

	# if dataset_description['dataset_type']=='classification':
	# 	with open(dataset_description['files_description'], mode='r') as csv_file:
	# 		csv_reader = csv.DictReader(csv_file)

	# 		for row in csv_reader:
	# 			tfr_paths=row['TFRecord Files']
	# 			file_path=row['VTK Files']
	# 			group=row['Group']

	# 			#extract data from tfrecord
	# 			print('extracting:',tfr_paths)
	# 			tfr_data,target=extractDataFromTFRecord(tfr_paths)

	# 			#extract data from shape
	# 			point_data,info = inputdata.load_features(file_path,feature_points=dataset_description['extracted_feature_points_info']['extraction_order'])
	# 			cell_data,info = inputdata.load_features(file_path,feature_polys=dataset_description['extracted_feature_polys_info']['extraction_order'])

	# 			data = np.concatenate((point_data,cell_data))

	# 			if (float(group) - target[0] != 0):
	# 				print('ERROR in groups :',group ,target)

	# 			for i in range(data.shape[0]):
	# 				if data[i]-tfr_data[i]>1.0e-07:
	# 					print('ERROR in data :',data[i]-tfr_data[i], i)

	# if dataset_description['dataset_type']=='generation':
	# 	with open(dataset_description['files_description'], mode='r') as csv_file:
	# 		csv_reader = csv.DictReader(csv_file)

	# 		for row in csv_reader:
	# 			tfr_paths=row['TFRecord Files']
	# 			infile_path=row['Input VTK Files']
	# 			outfile_path=row['Output VTK Files']


	# 			#extract data from tfrecord
	# 			print('extracting:',tfr_paths)
	# 			tfr_data,target=extractDataFromTFRecord(tfr_paths)

	# 			#extract data from shape
	# 			inpoint_data,info = inputdata.load_features(infile_path,feature_points=dataset_description['extracted_input_feature_points_info']['extraction_order'])
	# 			incell_data,info = inputdata.load_features(infile_path,feature_polys=dataset_description['extracted_input_feature_polys_info']['extraction_order'])

	# 			indata = np.concatenate((inpoint_data,incell_data))

	# 			outpoint_data,info = inputdata.load_features(outfile_path,feature_points=dataset_description['extracted_output_feature_points_info']['extraction_order'])
	# 			outcell_data,info = inputdata.load_features(outfile_path,feature_polys=dataset_description['extracted_output_feature_polys_info']['extraction_order'])

	# 			outdata = np.concatenate((outpoint_data,outcell_data))

	# 			for i in range(indata.shape[0]):
	# 				if indata[i]-tfr_data[i]>1.0e-07:
	# 					print('ERROR in input data :',indata[i]-tfr_data[i], i)

	# 			for i in range(outdata.shape[0]):
	# 				if outdata[i]-target[i]>1.0e-07:
	# 					print('ERROR in output data :',outdata[i]-tfr_data[i], i)


		








	
	# # SANITY CHECKS
	# print('dataset',np.shape(dataset))
	# print('labels',np.shape(labels))
	# print('dataset_res',np.shape(dataset_res))
	# print('labels_res',np.shape(labels_res))

	# print('num_train', num_train)
	# print('num_valid', num_valid)
	# print('number of labels',np.shape(np.unique(labels)))
	# print('number of labels resampled',np.shape(np.unique(labels_res)))
	# print('Labels resampled',np.unique(labels_res).tolist())

	








