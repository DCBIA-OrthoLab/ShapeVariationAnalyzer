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


# #############################################################################
# Create the data

parser = argparse.ArgumentParser(description='Multivariate Functional Shape Data Analysis (MFSDA)')
parser.add_argument('--shapeDir', type=str, help='Directory with vtk files .vtk', required=True)
parser.add_argument('--outputMean', help='output directory', default='mean.vtk')
parser.add_argument('--outputModel', help='output filename for model', default='model.pickle')

def readData(shapedir):
    """
    Run the commandline script for MFSDA.
    """
    """+++++++++++++++++++++++++++++++++++"""
    """Step 1. load dataset """

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

if __name__ == '__main__':
        
    args = parser.parse_args()

    start_all = timeit.default_timer()
    X, shapedata = readData(args.shapeDir)
    X_ = np.mean(X, axis=0, keepdims=True)
    print(X.shape, X_.shape)

    pointdata = X_.reshape(-1).reshape(-1, 3)
    polydatapoints = shapedata.GetPoints()
    ipoint = 0
    for point in pointdata:
        polydatapoints.SetPoint(ipoint, point[0], point[1], point[2])
        ipoint += 1

    writer = vtk.vtkPolyDataWriter()
    meanshapeoutputfilename = args.outputMean
    writer.SetFileName(meanshapeoutputfilename)
    writer.SetInputData(shapedata)
    writer.SetFileTypeToASCII()
    writer.Update()
    
    
    pca = PCA()
    pca.fit(X - X_)

    min_explained = 0.98
    sum_explained = 0.0
    num_components = 0
    
    for evr in pca.explained_variance_ratio_:
        sum_explained += evr
        num_components += 1
        if sum_explained >= min_explained:
            break
    
    print("num_components=",num_components)
    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X - X_)

    print(pca.explained_variance_ratio_)
    print(X_pca.shape)

    X_pca_ = np.mean(X_pca, axis=0, keepdims=True)
    X_pca_var = np.var(X_pca, axis=0, keepdims=True)

    pca_model = {}
    pca_model["pca"] = pca
    pca_model["X_"] = X_
    pca_model["X_pca_"] = X_pca_
    pca_model["X_pca_var"] = X_pca_var

    
    with open(args.outputModel, "wb") as outputfile:
        pickle.dump(pca_model, outputfile)
    
    stop_all = timeit.default_timer()
    delta_time_all = str(stop_all - start_all)
    print("The total elapsed time is " + delta_time_all)
