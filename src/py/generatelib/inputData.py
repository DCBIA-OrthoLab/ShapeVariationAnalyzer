import os
import numpy as np
import vtk
import pickle
import glob
import scipy.io as sio


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


    #
    #   Gets all the scalar array names in the polydata
    #
    def get_array_names(self, geometry):
        arraynames = []
        pointdata = geometry.GetPointData()
        for i in range(pointdata.GetNumberOfArrays()):
            arraynames.append(pointdata.GetArrayName(i))
        return np.array(arraynames)

    #
    # Function load_features(file)
    #   Load the shape stored in the filename "shape" and extract features (normals + mean distances + curvatures), stored in a 2D array (currentData)
    #   Features are normalized (normals are already done, in previous program SurfaceFeaturesExtractor with vtkPolyDataNormals)
    #
    def load_features(self, shape, feature_names = ["Normals", "Mean_Curvature", "distanceGroup"]):
        dataset = np.ndarray(shape=(1, self.NUM_POINTS, self.NUM_FEATURES), dtype=np.float32)

        try:
            print("Reading:", shape)
            reader_poly = vtk.vtkPolyDataReader()
            reader_poly.SetFileName(shape)
            # print "shape : " + shape

            reader_poly.Update()
            geometry = reader_poly.GetOutput()
            if not geometry.GetNumberOfPoints() == self.NUM_POINTS:
                raise Exception('Unexpected number of points in the shape: ' + str(geometry.GetNumberOfPoints()) + ' vs. ' + str(self.NUM_POINTS))
            
            extracted_feature_names = []
            features = []

            #Initialize an array with the same number of points
            for i in range(geometry.GetNumberOfPoints()):
                features.append([])

            #Get all array names
            arraynames = self.get_array_names(geometry)

            # --------------------------------- #
            # ----- GET ARRAY OF FEATURES ----- #
            # --------------------------------- #

            #Iterate over the featues we want to extract from the polydata
            for feature_name in feature_names:
                # Get the 'real names' of the scalar arrays by matching the features_name to the real array name in arraynames'
                reallarraynames = [name for name in arraynames if feature_name in name]

                #When the real scalar array name is extracted, we iterate over the real ones, i.e., if the extract_features_name is 'distanceGroup',
                #the array has ['distanceGroup0', 'distanceGroup1', ...]
                for arrayname in reallarraynames:
                    scalararray = geometry.GetPointData().GetScalars(arrayname)
                    if scalararray:
                        extracted_feature_names.append(arrayname)
                        for i in range(0, scalararray.GetNumberOfTuples()):
                            scalartup = scalararray.GetTuple(i)
                            features[i].extend(scalartup)

    
        except IOError as e:
            print('Could not read:', shape, ':', e, '- it\'s ok, skipping.')

        # print('Full dataset tensor:', dataset.shape)
        # print('Mean:', np.mean(dataset))
        # print('Standard deviation:', np.std(dataset))
        #print('new_feature_map',np.shape(new_feature_map))
        features = np.array(features)
        print('The following features were extracted', extracted_feature_names)
        print('features shape',np.shape(features))

        return features


    #
    # Function load_features_classe(folder, min_num_shapes)
    #   Call load_features for an entire folder/classe. Check if there's enough shapes in a classe.
    #

    def compute_laplace_beltrami(verts_coord,tris):
        ### computes a sparse matrix representing the discretized laplace-beltrami operator
        ### vertices: (num_points,3) array float
        ### tris: (num_triangles,3) array int 
        n = len(verts_coord)
        W_ij = np.empty(0)
        I = np.empty(0, np.int32)
        J = np.empty(0, np.int32)
        for i1, i2, i3 in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]: # for edge i2 --> i3 facing vertex i1
            vi1 = tris[:,i1] # vertex index of i1
            vi2 = tris[:,i2]
            vi3 = tris[:,i3]
            # vertex vi1 faces the edge between vi2--vi3
            # compute the angle at v1
            # add cotangent angle at v1 to opposite edge v2--v3
            # the cotangent weights are symmetric
            u = verts_coord[vi2] - verts_coord[vi1]
            v = verts_coord[vi3] - verts_coord[vi1]
            cotan = (u * v).sum(axis=1) / veclen(np.cross(u, v))
            W_ij = np.append(W_ij, 0.5 * cotan)
            I = np.append(I, vi2)
            J = np.append(J, vi3)
            W_ij = np.append(W_ij, 0.5 * cotan)
            I = np.append(I, vi3)
            J = np.append(J, vi2)
        L = sparse.csr_matrix((W_ij, (I, J)), shape=(n, n))
        # compute diagonal entries
        L = L - sparse.spdiags(L * np.ones(n), 0, n, n)
        L = L.tocsr()
        # area matrix
        e1 = verts_coord[tris[:,1]] - verts_coord[tris[:,0]]
        e2 = verts_coord[tris[:,2]] - verts_coord[tris[:,0]]
        n = np.cross(e1, e2)
        triangle_area = .5 * veclen(n)
        # compute per-vertex area
        vertex_area = np.zeros(len(verts_coord))
        ta3 = triangle_area / 3
        for i in xrange(tris.shape[1]):
            bc = np.bincount(tris[:,i].astype(int), ta3)
            vertex_area[:len(bc)] += bc
        VA = sparse.spdiags(vertex_area, 0, len(verts_coord), len(verts_coord))
        return L, VA


    def load_features_classe (self, vtklist, min_num_shapes=1, feature_names = ["Normals", "Mean_Curvature", "distanceGroup"]):
        # vtk_filenames = os.listdir(folder)  # Juste le nom du vtk file

        # Delete .DS_Store file if there is one
        # if vtk_filenames.count(".DS_Store"):
            # vtk_filenames.remove(".DS_Store")

        vtk_filenames = vtklist
        dataset = []
        # dataset = np.ndarray(shape=(len(vtk_filenames), self.NUM_POINTS, self.NUM_FEATURES), dtype=np.float32)
        # coordData_map = np.ndarray(shape=(len(vtk_filenames),self.NUM_POINTS,3),dtype=np.float32)
        for shape in vtk_filenames:
            # Prepare data
            features = self.load_features(shape, feature_names=feature_names)
            dataset.append(features)
        
        dataset = np.array(dataset)

        if np.shape(dataset)[0] < min_num_shapes:
            raise Exception('Fewer images than expected: %d < %d' % (num_shapes, min_num_shapes))

        #print('full feature map',feature_map)
        print('Full dataset tensor:', dataset.shape)
        print('Mean:', np.mean(dataset))
        print('Standard deviation:', np.std(dataset))
        print("")
        return dataset


    #
    # Function maybe_pickle(data_folders, min_num_shapes_per_class, force=False)
    #   Pickle features array sorted by class
    #
    def maybe_pickle(self, classFolders, min_num_shapes_per_class, force=False, feature_names = ["Normals", "Mean_Curvature", "distanceGroup"]):
        dataset_names = []
        # folders = list()
        # for d in data_folders:
        #     if os.path.isdir(os.path.join(data_folders, d)):
        #         folders.append(os.path.join(data_folders, d))

        #print('################')
        #mat_contents = sio.loadmat('shapes.mat',squeeze_me=True)
        #shape_matlab = mat_contents['shape']
        #shape1 = shape_matlab[0]
        #heat_kernel = shape1['sihks']
        #H = heat_kernel * 1
        #print('multiplie',H.shape)

        for classfolder in classFolders:

            set_filename = classfolder + '.pickle'

        # for folder in folders:
        #     head, tail = os.path.split(folder)
        #     set_filename = os.path.join(slicer.app.temporaryPath, tail + '.pickle')
            dataset_names.append(set_filename)
            if os.path.exists(set_filename) and not force:
                # You may override by setting force=True.
                print('%s already present - Skipping pickling.' % set_filename)
            else:

                vtklist = glob.glob(os.path.join(classfolder, "*.vtk"))
                print('Pickling %s.' % set_filename)
                dataset = self.load_features_classe(vtklist, min_num_shapes_per_class, feature_names=feature_names)

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

