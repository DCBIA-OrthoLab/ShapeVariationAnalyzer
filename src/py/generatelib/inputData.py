import os
import vtk
import pickle
import glob
import scipy.io as sio
import numpy as np

class inputData():
    def __init__(self, parent = None, num_points_param = 1002, num_classes_param = 7):
        if parent:
            parent.title = " "

        self.NUM_POINTS = num_points_param
        self.NUM_CLASSES = num_classes_param
        self.NUM_FEATURES = 3 + self.NUM_CLASSES + 4 +19  # Normals + NUM_CLASSES + curvatures


    #
    # Function get_folder_classes_list(datasetPath)
    # 	For a given folder, return the list of subfolders
    #
    def get_folder_classes_list(self, datasetPath):
        dataset_folders = [os.path.join(datasetPath, d) for d in sorted(os.listdir(datasetPath)) if os.path.isdir(os.path.join(datasetPath, d))]

        # Delete .DS_Store file if there is one
        if dataset_folders.count(str(datasetPath) + ".DS_Store"):
            dataset_folders.remove(str(datasetPath) + ".DS_Store")

        return dataset_folders

    def get_vtklist(self, classFolders):

        vtk_list_classes = {}

        for classfolder in classFolders:
            vtk_list_classes[classfolder] = glob.glob(os.path.join(classfolder, "*.vtk"))

        return vtk_list_classes
    #
    #   Gets all the scalar array names in the polydata for the points
    #
    def get_points_array_names(self, geometry):
        arraynames = []
        pointdata = geometry.GetPointData()
        for i in range(pointdata.GetNumberOfArrays()):
            arraynames.append(pointdata.GetArrayName(i))
        return np.array(arraynames)

    #
    #   Gets all the scalar array names in the polydata for polys
    #
    def get_polys_array_names(self, geometry):
        arraynames = []
        celldata = geometry.GetCellData()
        for i in range(celldata.GetNumberOfArrays()):
            arraynames.append(celldata.GetArrayName(i))
        return np.array(arraynames)

    #
    # Function load_features(file)
    #   Load the shape stored in the filename "shape" and extract features (normals + mean distances + curvatures), stored in a 2D array (currentData)
    #   Features are normalized (normals are already done, in previous program SurfaceFeaturesExtractor with vtkPolyDataNormals)
    #
    def load_features(self, shape, feature_points = None, feature_polys = None):

        try:
            print("Reading:", shape)
            reader_poly = vtk.vtkPolyDataReader()
            reader_poly.SetFileName(shape)
            # print "shape : " + shape

            reader_poly.Update()
            geometry = reader_poly.GetOutput()
            if not geometry.GetNumberOfPoints() == self.NUM_POINTS:
                raise Exception('Unexpected number of points in the shape: ' + str(geometry.GetNumberOfPoints()) + ' vs. ' + str(self.NUM_POINTS))

            if feature_points and feature_polys:
                print("WARNING!!! You have set both feature_points and feature_polys, extracting feature_points only!!!")
            
            extracted_feature_points = []
            features = []

            if feature_points:

                #Initialize an array with the same number of points
                for i in range(geometry.GetNumberOfPoints()):
                    features.append([])

                #Get all array names
                arraynames = self.get_points_array_names(geometry)

                # --------------------------------- #
                # ----- GET ARRAY OF FEATURES ----- #
                # --------------------------------- #

                #Iterate over the featues we want to extract from the polydata
                for feature_name in feature_points:

                    if feature_name == "Points":
                        extracted_feature_points.append("Points")

                        for i in range(geometry.GetNumberOfPoints()):
                            scalartup = geometry.GetPoint(i)
                            features[i].extend(scalartup)
                    # Get the 'real names' of the scalar arrays by matching the features_name to the real array name in arraynames'
                    else:
                        reallarraynames = [name for name in arraynames if feature_name in name]

                        #When the real scalar array name is extracted, we iterate over the real ones, i.e., if the extract_features_name is 'distanceGroup',
                        #the array has ['distanceGroup0', 'distanceGroup1', ...]
                        for arrayname in reallarraynames:
                            scalararray = geometry.GetPointData().GetScalars(arrayname)
                            if scalararray:
                                extracted_feature_points.append(arrayname)
                                for i in range(0, scalararray.GetNumberOfTuples()):
                                    scalartup = scalararray.GetTuple(i)
                                    features[i].extend(scalartup)
                
                features = np.array(features)
                print('\tThe following features were extracted', extracted_feature_points)
                print('\tfeatures shape',np.shape(features))

                return features

            elif feature_polys:

                polys = geometry.GetPolys()

                pointidlist = vtk.vtkIdList()
                features = []

                #Get all array names
                arraynames = self.get_polys_array_names(geometry)

                for feature_name in feature_polys:
                    cellfeatures = []
                    
                    if feature_name == "Points":
                        extracted_feature_points.append("Points")

                        for ci in range(geometry.GetNumberOfPolys()):
                            geometry.GetCellPoints(ci, pointidlist)

                            pointidlisttemp = vtk.vtkIdList()
                            cellidlisttemp = vtk.vtkIdList()

                            print("cell dd")
                            geometry.GetCellEdgeNeighbors(ci, pointidlist.GetId(0), pointidlist.GetId(1), cellidlisttemp)
                            print("cell dd111")
                            print("cell Numids", cellidlisttemp.GetNumberOfIds())

                            npfeatures = []

                            for pid in range(pointidlist.GetNumberOfIds()):
                                point = geometry.GetPoint(pointidlist.GetId(pid))
                                npfeatures.append([p for p in point])

                            cellfeatures.append(npfeatures)
                            
                    # Get the 'real names' of the scalar arrays by matching the features_name to the real array name in arraynames'
                    else:
                        reallarraynames = [name for name in arraynames if feature_name in name]

                        #When the real scalar array name is extracted, we iterate over the real ones, i.e., if the extract_features_name is 'distanceGroup',
                        #the array has ['distanceGroup0', 'distanceGroup1', ...]
                        for arrayname in reallarraynames:
                            scalararray = geometry.GetCellData().GetScalars(arrayname)
                            if scalararray:
                                extracted_feature_points.append(arrayname)
                                for ci in range(geometry.GetNumberOfPolys()):

                                    scalartup = scalararray.GetTuple(ci)
                                    cellfeatures.append(scalartup)

                    features.append(cellfeatures)

                print('\tThe following features were extracted', extracted_feature_points)
                print('\tfeatures shape',np.shape(features))
                return features

            else:
                raise Exception('You must set one of feature_polys or feature_polys to extract data from the shape')

        except IOError as e:
            print('Could not read:', shape, ':', e, '- it\'s ok, skipping.')



    def load_features_class(self, vtklist, min_num_shapes=1, feature_points = None, feature_polys = False):

        vtk_filenames = vtklist
        dataset = []
        
        for shape in vtk_filenames:
            # Prepare data
            features = self.load_features(shape, feature_points=feature_points, feature_polys = feature_polys)
            if features is not None:
                dataset.append(features)
        
        dataset = np.array(dataset)

        if np.shape(dataset)[0] < min_num_shapes:
            raise Exception('Fewer samples than expected: %d < %d' % (num_shapes, min_num_shapes))
        
        return dataset

    #
    # Function maybe_pickle(data_folders, min_num_shapes_per_class, force=False)
    #   Pickle features array sorted by class
    #
    def maybe_pickle(self, classFolders, min_num_shapes_per_class, force=False, feature_points = None, feature_polys = False):
        
        dataset_names = []
        vtkdict = self.get_vtklist(classFolders)

        for classfolder, vtklist in vtkdict.items():

            set_filename = classfolder + '.pickle'
            dataset_names.append(set_filename)

            if os.path.exists(set_filename) and not force:
                # You may override by setting force=True.
                print('%s already present - Skipping pickling.' % set_filename)
            else:

                print('Pickling %s.' % set_filename)
                dataset = self.load_features_class(vtklist, min_num_shapes_per_class, feature_points=feature_points, feature_polys=feature_polys)

                print(np.shape(dataset))
                try:
                    with open(set_filename, 'wb') as f:
                        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print('Unable to save data to', set_filename, ':', e)
        
        
        return dataset_names

    #
    # Function randomize(dataset, labels)
    #   Randomize the data and their labels
    #
    def randomize(self,dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation]

        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels

