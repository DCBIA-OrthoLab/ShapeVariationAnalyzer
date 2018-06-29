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

#from scipy import stats
import vtk
import os
import argparse
import timeit
import cPickle as pickle
import random
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pprint 
import inputData
#from sklearn.decomposition import PCA
import math
import inputData
import glob
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score


#from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, roc_curve, auc
import itertools
from sklearn import preprocessing



# #############################################################################
# Generate data

parser = argparse.ArgumentParser(description='Shape Variation Analyzer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--model', type=str, help='pickle file with the pca decomposition', required=True)
#parser.add_argument('--shapeDir', type=str, help='Directory with vtk files .vtk', required=True)
parser.add_argument('--picklefile',dest='picklefile',help='picklefile with the dataset',required=True)
#parser.add_argument('--dataPathtrain', action='store', dest='dirwithSubtrain', help='folder with subclasses', required=True)
#parser.add_argument('--dataPathtest', action='store', dest='dirwithSubtest', help='folder with subclasses', required=True)
parser.add_argument('--train_size', help='train ratio', type=float, default=0.8)
parser.add_argument('--validation_size', help='validation ratio from test data', default=0.5, type=float)
parser.add_argument('--feature_names', help='Extract the following features from the polydatas', nargs='+', default=["Normals", "Mean_Curvature", "distanceGroup"], type=str)
parser.add_argument('--out', dest="pickle_file_new", help='Pickle file output', default="new_dataset.pickle", type=str)

#parser.add_argument('-outputdataPath', action='store', dest='dirwithSubGenerated', help='folder with subclasses after generation of data', required=True)
#parser.add_argument('--outputGenerated', help='output folder for shapes', default='./out')
#parser.add_argument('--num_shapes', type=int, help='number shapes to be generated', default=10)
#parser.add_argument('--meanShape',help='mean shape', required=True)

class generation_shapes:
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
						print('num shapes per group',label,num_shapes_per_group)
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


	def plot_confusion_matrix(cm, classes,
	                          normalize=False,
	                          title='Confusion matrix',
	                          cmap=plt.cm.Blues):
	    
	    """
	    This function prints and plots the confusion matrix.
	    Normalization can be applied by setting `normalize=True`.
	    """
	    if normalize:
	        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	        print("Normalized confusion matrix")
	    else:
	        print('Confusion matrix, without normalization')

	    print(cm)
	    plt.figure()
	    plt.imshow(cm, interpolation='nearest', cmap=cmap)
	    plt.title(title)
	    plt.colorbar()
	    tick_marks = np.arange(len(classes))
	    plt.xticks(tick_marks, classes, rotation=45)
	    plt.yticks(tick_marks, classes)

	    fmt = '.2f' if normalize else 'd'
	    thresh = cm.max() / 2.
	    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	        plt.text(j, i, format(cm[i, j], fmt),
	                 horizontalalignment="center",
	                 color="white" if cm[i, j] > thresh else "black")

	    #plt.tight_layout()
	    plt.ylabel('True label')
	    plt.xlabel('Predicted label')

	def training_acc(X_train,y_train,X_test,y_test,classifiers):
		
		
		tab_score=[]
		for clf in classifiers:
			clf.fit(X_train, y_train)
			score = clf.score(X_test, y_test)
			print('score',score)
			tab_score.append(score)
		
		print('TAB score',tab_score)
		return tab_score



	def SVM_classification(X_dataset,y_labels,dataset_test,labels_test):
		

		# model = svm.SVC(decision_function_shape='ovr',kernel='rbf',C=100,gamma=10)
		# model.fit(X_dataset,y_labels)
		# model.score(X_dataset,y_labels)

		print('data shape',X_dataset.shape)
		
		
		# predicted_labels = model.predict(test_dataset)
		# acc = accuracy_score(test_labels,predicted_labels,normalize=True)
		# print('accuracy',acc)

		h = .02  # step size in the mesh

		names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
	         "Decision Tree", "Random Forest", "MLP Classifier", "AdaBoost",
	         "Naive Bayes", "QDA"]

		classifiers = [
	    KNeighborsClassifier(3),
	    SVC(kernel="linear", C=0.025),
	    SVC(gamma=2, C=1),
	    GaussianProcessClassifier(1.0 * RBF(1.0)),
	    DecisionTreeClassifier(max_depth=5),
	    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
	    MLPClassifier(alpha=1),
	    AdaBoostClassifier(),
	    GaussianNB(),
	    QuadraticDiscriminantAnalysis()]

		#linearly_separable = (X_dataset,y_labels)
		#datasets = [linearly_separable]
	 	

	 	i=1
	 		#standardizing features
	 		#X_stand = StandardScaler().fit_transform(X)
	 	X_train,X_test, y_train, y_test = train_test_split(X_dataset,y_labels, test_size =0.4)
		    # just plot the dataset first
		cm = plt.cm.RdBu
			#cm_bright = ListedColormap(['#FF0000', '#0000FF','#48FF00'])

		#fig1,ax1=plt.subplots(3,4)
		#fig2,ax2=plt.subplots(3,4)
			#if ds_cnt == 0:
			#	ax.set_title("Input data")
		    # Plot the training points

			#score=training_acc(X_train,y_train,X_test,y_test,classifiers)
		    # iterate over classifiers
		#for name, clf in zip(names, classifiers):
		#	ax = plt.subplot(3, 4, i)	
			
		#	clf.fit(X_train, y_train)
				#score = clf.score(X_test[:,2:], y_test)
				#print('score 2 features',score)
				#make meshgrid
		#	x_min, x_max = X_train[:, 0].min()-1, X_train[:, 0].max()+1
		#	y_min, y_max = X_train[:, 1].min()-1, X_train[:, 1].max()+1

				# xx, yy= np.meshgrid(np.arange(x_min, x_max,1) ,np.arange(y_min, y_max, 1))

				# print('shape xx',xx.shape,'shape yy',yy.shape)
				# print('shape ravel',np.c_[xx.ravel(),yy.ravel()].shape)

				# if hasattr(clf, "decision_function"):
				# 	Z = clf.decision_function(np.c_[xx.ravel(),yy.ravel()])
				# 	print('xx shape',xx.shape,'yy shape',yy.shape,'Z shape',Z.shape)
				# 	Z=Z[:,1]
				# else:
				# 	Z = clf.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1]
				# 	print('Z shape',Z.shape)
				# Z = Z.reshape(xx.shape)
				# ax.contourf(xx,yy,Z, cmap=plt.cm.get_cmap('nipy_spectral',np.shape(np.unique(y_test))[0]), alpha=.4)


		        # Plot also the training points
		#	CS=ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.get_cmap('nipy_spectral',np.shape(np.unique(y_train))[0]))
		        # and testing points
		#	ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.get_cmap('nipy_spectral',np.shape(np.unique(y_test))[0]),edgecolors='k', alpha=0.6)
				
		#	ax.set_xticks(())
		#	ax.set_yticks(())

		#	ax.set_title(name)
				#plt.suptitle(score,y=1.05,fontsize=18)
				#a=0.98
				
				#if name=="Current neural network":
				#	ax.text(240, -40, a,size=15, horizontalalignment='right')

				#else:
				#ax.text(240, -40, ('%.2f' % score).lstrip('0'),size=15, horizontalalignment='right')
				#clf.fit(X_train, y_train)
				#score_training = clf.score(X_train, y_train)
				#print('score training',score_training)
				#ax.text(x_max-0.5, y_min+0.5, ('%.2f' % score).lstrip('0'),size=15, horizontalalignment='right')

		for name,clf in zip(names,classifiers):
			clf.fit(X_train, y_train)
			y_prediction = clf.predict(dataset_test)
			
			

			print('y_prediction',y_prediction,'labels_test',labels_test)
			test_score = accuracy_score(labels_test,y_prediction)
			confusion = confusion_matrix(labels_test,y_prediction)
			print('The accuracy of ',name,'is',test_score)
			name_labels=["group0","group1","group2","group3","group4","group5"]
			plot_confusion_matrix(confusion,name_labels,title=name)

			if hasattr(clf, "decision_function"):
				#binarize labels
				lb = preprocessing.LabelBinarizer()
				lb.fit([0,1,2,3,4,5])
				print('y_test',y_test)
				y_test_bin=lb.transform(y_test)
				fpr=dict()
				tpr=dict()
				roc_auc=dict()
				y_score = clf.fit(X_train,y_train).decision_function(X_test)
				print('y_score',y_score)
				print('y_score shape',y_score.shape,'y_test shape',y_test.shape)
				#compute ROC curve and ROC area for each class
				for j in range(6):
					print(j)
					fpr[j], tpr[j], _ = roc_curve(y_test_bin[:,j],y_score[:,j])
					roc_auc[j]=auc(fpr[j],tpr[j])

				plt.figure()
				lw=2
				plt.plot(fpr[2],tpr[2],color='darkorange',lw=lw,label='ROC curve (area = %0.2f)'%roc_auc[2])
				plt.show
				#ax.text(x_max-1,y_min+0.1,('%.3f' % score_training),size=10,horizontalalignment='right')
				#score=trainin_acc(X_train,y_train,X_test,y_test,classifiers)
				#ax.text(x_max-0.5,y_min+0.1,('%.3f' % score[i-1]),size=8,horizontalalignment='right')

			print('score testing',test_score)
			#plt.colorbar(CS)



			i += 1

		#printing our neural network accuracy
		# ax=plt.subplot(3,4,11)
		# ax.scatter(X_dataset[:, 0], X_dataset[:, 1], c=y_labels, cmap=plt.cm.get_cmap('nipy_spectral',np.shape(np.unique(y_labels))[0]))
		# ax.scatter(dataset_test[:, 0], dataset_test[:, 1], c=labels_test, cmap=plt.cm.get_cmap('nipy_spectral',np.shape(np.unique(labels_test))[0]),edgecolors='k', alpha=0.6)
		# ax.set_xticks(())
		# ax.set_yticks(())
		# ax.set_title("5-layers neural network")
		# score_fake=0.971
		# maxi=X_dataset[:,0].max()
		# mini=X_dataset[:,0].min()
		# maxi1=X_dataset[:,1].max()
		# mini1=X_dataset[:,1].min()

		# print('x_max',maxi,'x_min',mini,'y_max',maxi1,'y_min',mini1)
		# #plt.axis([0,1,-1,1])
		# ax.text(maxi,mini1,('%.3f' % score_fake),size=8,horizontalalignment='right')



		plt.tight_layout()
		plt.show()
	def generate(args):
		np.set_printoptions(threshold='nan')

		print('###########In generation shape#############')
		#
		#dataPathtrain=args.dirwithSubtrain
		#dataPathtest=args.dirwithSubtest
		pickle_file = args.picklefile
		pickle_file_output= args.pickle_file_new

		# Get the data from the folders with vtk files
		inputdata = inputData.inputData()
		fi = open(pickle_file,'rb')
		dataset=pickle.load(fi)
		test_labels =dataset["test_labels"]
		train_labels =dataset["train_labels"]
		valid_labels =dataset["valid_labels"]
		test_dataset =dataset["test_dataset"]
		train_dataset =dataset["train_dataset"]
		valid_dataset =dataset["valid_dataset"]


		print(test_labels)

		#data_folders_train = inputdata.get_folder_classes_list(dataPathtrain)
		#data_folders_test = inputdata.get_folder_classes_list(dataPathtest)
		#pickled_datasets_train,vtklisttrain = inputdata.maybe_pickle(data_folders_train, 6, feature_points=args.feature_names)
		#pickled_datasets_test,vtklisttest = inputdata.maybe_pickle(data_folders_test, 0, feature_points=args.feature_names)


		#Create the labels, i.e., enumerate the groups
		#dataset_train,labels_train = get_labels(pickled_datasets_train)
		#print('pickled_datasets_train',pickled_datasets_train,'pickled_datasets_test',pickled_datasets_test)
		#dataset_test,labels_test = get_labels(pickled_datasets_test)



		# Compute the total number of shapes and train/test size
		total_number_shapes_train=train_dataset.shape[0]
		total_number_shapes_test=test_dataset.shape[0]
		print('total number of shapes train',total_number_shapes_train)
		print('total number of shapes test', total_number_shapes_test)
		print('labels to train',train_labels,'labels to test',test_labels)
		#num_train = int(args.train_size*total_number_shapes_train)
		#num_valid = int((total_number_shapes_train - num_train)*args.validation_size)

		# Randomize the original dataset
		#print('shape before randomize',dataset_train.shape)
		shuffled_dataset, shuffled_labels = inputdata.randomize(train_dataset, train_labels)
		#print('shape after randomize',shuffled_dataset.shape)
		#shuffled_dataset_test,shuffled_labels_test = inputdata.randomize(dataset_test,labels_test)

		#shuffled_dataset = np.reshape(shuffled_dataset, (total_number_shapes_train, -1))
		#print('shape after reshape',shuffled_dataset.shape)
		#shuffled_dataset_test = np .reshape(shuffled_dataset_test,(total_number_shapes_test,-1))

		# Generate SMOTE with out including the valid/test samples, in some cases, this may raise an error
		# as the number of samples in one class is less than 5 and SMOTE cannot continue. Just run it again
		dataset_res,labels_res=generate_with_SMOTE(shuffled_dataset,shuffled_labels)

		
		# SANITY CHECKS
		print('dataset train',np.shape(train_dataset))
		print('labels train',np.shape(train_labels))
		#print('dataset_res',np.shape(dataset_res))
		#print('labels_res',np.shape(labels_res))

		#print('num_train', num_train)
		#print('num_valid', num_valid)
		print('number of labels',np.shape(np.unique(train_labels)))
		#print('number of labels resampled',np.shape(np.unique(labels_res)))
		#print('Labels resampled',np.unique(labels_res).tolist())
		print('test labels', test_labels)


		#SVM_classification(dataset_res,labels_res,dataset_test,labels_test)

		
		#clf=LinearSVC(random_state=0)
		#clf=GaussianProcessClassifier(1.0 * RBF(1.0))

		#clf.fit(dataset_res,labels_res)
		#prediction = clf.predict(dataset_test)


		#for i in range(0,total_number_shapes_test):
		#	head,tail = os.path.split(vtklisttest[i])
		#	print(tail,prediction[i])

		#PCA_plot(dataset,labels,dataset_res,labels_res)

		try:

			f = open(pickle_file_output, 'wb')
			
			save = {
	        #'train_dataset': dataset_res,
	        #'train_labels': labels_res,
	        'train_dataset': dataset_res,
	        'train_labels': labels_res,
			'valid_dataset': valid_dataset,
	        'valid_labels': valid_labels,
	       # 'test_dataset': dataset[num_train + num_valid:],
	       #'test_labels': labels[num_train + num_valid:]        
	  		'test_dataset': test_dataset,
	        'test_labels': test_labels 
			}
			pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	    	#f.close()
			
		except Exception as e:
			print('Unable to save data to', pickle_file, ':', e)
			raise


#if __name__ == '__main__':
#	args = parser.parse_args()
#	generate(args)

