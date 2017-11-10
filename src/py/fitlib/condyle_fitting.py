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
import scipy.optimize as optimize
from scipy.optimize import least_squares
import vtk
import os
import argparse
import timeit
import pickle


# #############################################################################
# Create the data

parser = argparse.ArgumentParser(description='Multivariate Functional Shape Data Analysis (MFSDA)')
parser.add_argument('--model', type=str, help='pickle file with the pca decomposition', required=True)
parser.add_argument('--mean', type=str, help='mean shape', required=True)
parser.add_argument('--shape', help='shape to fit the model', required=True)
parser.add_argument('--output', type=str, help='output model fitted', default='output.vtk')

pca = 0
pointlocator = 0
shapepoints = 0
X_ = 0

def cost_fit(x):

    points = pca.inverse_transform(x.reshape(1, -1))
    points += X_
    points = points.reshape(-1).reshape(-1, 3)
    
    costdiff = []
    for p in points:
        pid = pointlocator.FindClosestPoint(p)
        shapepoint = shapepoints.GetPoint(pid)
        costdiff.append(p - shapepoint)
    
    return np.array(costdiff).reshape(-1)

if __name__ == '__main__':
        
    args = parser.parse_args()

    start_all = timeit.default_timer()

    print("Reading mean", args.mean)
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(args.mean)
    reader.Update()
    meanshape = reader.GetOutput()

    print("Reading shape", args.shape)
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(args.shape)
    reader.Update()
    shape = reader.GetOutput()

    # pca_model = {}
    # pca_model["pca"] = pca
    # pca_model["X_"] = X_
    # pca_model["X_pca_"] = X_pca_
    # pca_model["X_pca_var"] = X_pca_var
    
    print("Reading pca model", args.model)
    with open(args.model, "rb") as inputmodel:
        pca_model = pickle.load( inputmodel )


    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(meanshape)
    icp.SetTarget(shape)
    icp.SetCheckMeanDistance(1)
    icp.SetMaximumMeanDistance(0.001)
    icp.SetMaximumNumberOfIterations(10000)
    icp.SetMaximumNumberOfLandmarks(500)
    icp.GetLandmarkTransform().SetModeToAffine()
    icp.SetMeanDistanceModeToRMS()
    icp.SetStartByMatchingCentroids(True)

    transformpolydata = vtk.vtkTransformPolyDataFilter()
    transformpolydata.SetInputData(meanshape)
    transformpolydata.SetTransform(icp)
    transformpolydata.Update()
    initialshape = transformpolydata.GetOutput()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(args.output)
    writer.SetInputData(initialshape)
    writer.SetFileTypeToASCII()
    writer.Update()


    initialshapedatapoints = initialshape.GetPoints()
    initpoints = []
    for i in range(initialshapedatapoints.GetNumberOfPoints()):
        p = initialshapedatapoints.GetPoint(i)
        initpoints.append(p)

    initpoints = np.array(initpoints)
    initpoints = initpoints.reshape(1 ,-1)

    pca = pca_model["pca"]
    X_ = pca_model["X_"]
    initcoeff = pca.transform(initpoints - X_)

    shapepoints = shape.GetPoints()
    pointlocator = vtk.vtkPointLocator()
    pointlocator.SetDataSet(shape)
    pointlocator.BuildLocator()

    print(initcoeff)
    res = least_squares(cost_fit, initcoeff.reshape(-1), loss='soft_l1')
    #res = least_squares(cost_fit, initcoeff.reshape(-1), loss='cauchy')
    print(res.x)
    res_points = pca.inverse_transform(res.x.reshape(1, -1))
    res_points += X_
    res_points = res_points.reshape(-1).reshape(-1, 3)

    meanshapepoints = meanshape.GetPoints()
    meani = 0
    for p in res_points:
        meanshapepoints.SetPoint(meani, p)
        meani += 1

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(args.output)
    writer.SetInputData(meanshape)
    writer.SetFileTypeToASCII()
    writer.Update()
    
    stop_all = timeit.default_timer()
    delta_time_all = str(stop_all - start_all)
    print("The total elapsed time is " + delta_time_all)
