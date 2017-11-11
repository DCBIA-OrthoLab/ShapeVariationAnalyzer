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

# #############################################################################
# Generate data

parser = argparse.ArgumentParser(description='Multivariate Functional Shape Data Analysis (MFSDA)')
parser.add_argument('--model', type=str, help='pickle file with the pca decomposition', required=True)
parser.add_argument('--outputGenerated', help='output folder for shapes', default='./out')
parser.add_argument('--num_shapes', type=int, help='number shapes to be generated', default=10)
parser.add_argument('--meanShape',help='mean shape', required=True)


def generate_data(pca_model):

	print("Generating data ...")

	pca = pca_model["pca"]
	X_ = pca_model["X_"] 
	X_pca_ = pca_model["X_pca_"]
	X_pca_var = pca_model["X_pca_var"] 
	print('Variance',X_pca_var)
	print('Mean',X_pca_)

	#between -1 and 1
	alpha=0.5*(np.random.random_sample(np.size(X_pca_)))-1
	print(alpha)

	data_compressed = 2 * X_pca_var * alpha + X_pca_
	print('data compressed',data_compressed)
	print('shape data compressed',np.shape(data_compressed))
	data_generated = pca.inverse_transform(data_compressed) + X_

	print('data generated',data_generated)

	return data_generated



if __name__ == '__main__':

	args = parser.parse_args()

	with open(args.model,"rb") as PCAfile :
		pca_model=pickle.load(PCAfile)


	reader = vtk.vtkPolyDataReader()
	reader.SetFileName(args.meanShape)
	reader.Update()
	shapedata = reader.GetOutput()
	shapedatapoints = shapedata.GetPoints()

	for i in range(args.num_shapes) :
		data_generated = generate_data(pca_model)
		pointdata = data_generated.reshape(-1).reshape(-1, 3)
		ipoint = 0

		for point in pointdata:
			shapedatapoints.SetPoint(ipoint, point[0], point[1], point[2])
	    	ipoint += 1

		print('Writing files...')
		writer = vtk.vtkPolyDataWriter()
		filename=os.path.join(args.outputGenerated,str(i)+'.vtk')
		writer.SetFileName(os.path.join(args.outputGenerated,str(i)+'.vtk'))
		writer.SetInputData(shapedata)
		writer.SetFileTypeToASCII()
		writer.Update()
    	print(filename)
    




