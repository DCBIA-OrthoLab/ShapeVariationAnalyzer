import os
import vtk
import pickle
import glob
import scipy.io as sio
import numpy as np
import json
import csv
from collections import OrderedDict
import tensorflow as tf

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

        vtk_list_classes = OrderedDict()
        
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
            extracted_feature_info=dict()

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

                        points=[]
                        for i in range(geometry.GetNumberOfPoints()):
                            scalartup = geometry.GetPoint(i)
                            features[i].extend(scalartup)

                        extracted_feature_info['Points']=dict()
                        extracted_feature_info['Points']['length']=len(scalartup)
                        
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
                                extracted_feature_info[arrayname]=dict()
                                extracted_feature_info[arrayname]['length']=len(scalartup)

                extracted_feature_info['number_of_points']=geometry.GetNumberOfPoints()
                extracted_feature_info['extraction_order']=extracted_feature_points
                features = np.array(features)
                features=features.reshape(-1)
                print('\tThe following features were extracted', extracted_feature_points)
                print('\tfeatures shape',np.shape(features))

                return features , extracted_feature_info

            elif feature_polys:

                polys = geometry.GetPolys()

                pointidlist = vtk.vtkIdList()
                #Initialize an array with the same number of points
                for i in range(geometry.GetNumberOfCells()):
                    features.append([])

                #Get all array names
                arraynames = self.get_polys_array_names(geometry)

                for feature_name in feature_polys:
                    
                    if feature_name == "Points":
                        extracted_feature_points.append("Points")

                        for ci in range(geometry.GetNumberOfPolys()):
                            geometry.GetCellPoints(ci, pointidlist)

                            scalartup=[]
                            for pid in range(pointidlist.GetNumberOfIds()):
                                point = geometry.GetPoint(pointidlist.GetId(pid))
                                scalartup.extend(point)

                            features[ci].extend(scalartup)


                        extracted_feature_info['Points']=dict()
                        extracted_feature_info['Points']['length']=len(scalartup)
                            
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
                                    features[ci].extend(scalartup)
                                extracted_feature_info[arrayname]=dict()
                                extracted_feature_info[arrayname]['length']=len(scalartup)

                extracted_feature_info['number_of_cells']=geometry.GetNumberOfCells()
                extracted_feature_info['extraction_order']=extracted_feature_points
                features = np.array(features)
                features=features.reshape(-1)
                print('\tThe following features were extracted', extracted_feature_points)
                print('\tfeatures shape',np.shape(features))
                return features ,extracted_feature_info

            else:
                raise Exception('You must set one of feature_polys or feature_polys to extract data from the shape')

        except IOError as e:
            print('Could not read:', shape, ':', e, '- it\'s ok, skipping.')



    def load_features_class(self, vtklist, min_num_shapes=1, feature_points = None, feature_polys = None):

        vtk_filenames = vtklist
        dataset = []

        for shape in vtk_filenames:
            # Prepare data
            features ,extracted_feature_info = self.load_features(shape, feature_points=feature_points, feature_polys = feature_polys)
            if features is not None:
                dataset.append(features)
        
        dataset = np.array(dataset)

        if np.shape(dataset)[0] < min_num_shapes:
            raise Exception('Fewer samples than expected: %d < %d' % (np.shape(dataset)[0], min_num_shapes))
        
        return dataset, extracted_feature_info

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

            description_path=os.path.join(os.path.dirname(set_filename),'extraction_description.json')

            if os.path.exists(set_filename) and not force:
                # You may override by setting force=True.
                print('%s already present - Skipping pickling.' % set_filename)
            else:

                print('Pickling %s.' % set_filename)
                dataset,extracted_feature_info = self.load_features_class(vtklist, min_num_shapes_per_class, feature_points=feature_points, feature_polys=feature_polys)

                try:
                    with open(set_filename, 'wb') as f:
                        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print('Unable to save data to', set_filename, ':', e)


                
                try:
                    with open(description_path, 'w') as f:
                        json.dump(extracted_feature_info, f,indent = 4)
                except Exception as e:
                    print('Unable to save extraction description to', description_path, ':', e)
        
        
        return dataset_names , description_path

    #
    # Function randomize(dataset, labels)
    #   Randomize the data and their labels
    #
    def randomize(self,dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation]

        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels

    def writeTFRecord(self,feature_data,path):
    #write a TFRecord at the defined path containing the features

        #convert features in the correct format
        for feature_name , value in feature_data.items():
            if value.dtype == 'float64':
                feature_data[feature_name]=self._float_feature(value.tolist())
            elif value.dtype == 'int64':
                feature_data[feature_name]=self._int64_feature(value.tolist())
            else :
                feature_data[feature_name]=self._bytes_feature(value.tolist())

        #Create a tensorflow global feature containing all the features
        features_data = tf.train.Features(feature=feature_data)

        #create the tensorflow example
        features_example=tf.train.Example(features=features_data)

        #write the example in a tfRecord
        try:
            os.mkdir(os.path.dirname(path))
        except:
            pass
        with tf.python_io.TFRecordWriter(path) as writer:
            writer.write(features_example.SerializeToString())
        print('Saved record: '+path)
        
    def writeRecords(self,record_dir,input_features,target_features,start_id=0,file_name_prefix='TFR_'):
    #write all TFRecords in the dataset_path in a folder named dir_name
    #each TFRecord contain a row of input_features and his associated row in target_features

        #create folder
        try:
            os.mkdir(os.path.dirname(record_dir))
        except:
            pass

        record_list=[]

        for i in range(input_features.shape[0]):
            #extract the features
            feature_dict=dict()
            feature_dict['input']=input_features[i,:]

            if len(target_features.shape)==1:
                feature_dict['output']=target_features[i]
            if len(target_features.shape)==2:
                feature_dict['output']=target_features[i,:]

            #generate file name
            record_path=os.path.join(record_dir,file_name_prefix+str(start_id+i)+'.tfrecord')
            self.writeTFRecord(feature_dict,record_path)

            record_list.append(record_path)

        return record_list

    #numpy array to tfrecord conversion
    def _int64_feature(self,value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    def _float_feature(self,value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    def _bytes_feature(self,value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def TFRecord_dataset(self, csvFile,records_path, min_num_shapes_per_class=5, force=False, feature_points = None, feature_polys = None,feature_points_output = None,feature_polys_output=None):
        #read CSV file, determine if classification or generation dataset
        with open(csvFile, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            column_names = []
            
            dataset_type=None

            for row in csv_reader:
                #get columns names and #identify dataset type
                if len(column_names)==0:
                    for name,v in row.items():
                        column_names.append(name)                   
                    if 'VTK Files' in column_names and 'Group' in column_names:
                        dataset_type='classification'
                        print('dataset type : '+ dataset_type)
                        class_names=[]
                        dict_vtkfiles=dict()
                    elif 'Input VTK Files' in column_names and 'Output VTK Files' in column_names:
                        dataset_type='generation'
                        print('dataset type : '+ dataset_type)
                        vtkfiles_in=[]
                        vtkfiles_out=[]
                    else:
                        print('Impossible to determine the dataset type!')
                        print("For classification type, the csv file should contain a row named 'VTK Files' and 'Group'.")
                        print("For generation type, the csv file should contain a row named 'input VTK File' and 'output VTK File'.")
                        raise Exception('Unknown dataset type')
                        return  

                if dataset_type=='classification':
                    group=row['Group']
                    vtkfile=row['VTK Files']
                    if group not in class_names:
                        class_names.append(group)
                        dict_vtkfiles[group]=[]
                    dict_vtkfiles[group].append(vtkfile)


                elif dataset_type=='generation':
                    vtkfiles_in.append(row['Input VTK Files'])
                    vtkfiles_out.append(row['Output VTK Files'])
                    


                    
        if dataset_type=='classification':
            #associating numbers to class_names
            #class correspondance
            class_corres=dict()
            for label, name in enumerate(class_names):
                class_corres[name]=label

            #extract features for each class 
            ##check for existing features
            if (feature_points == None and feature_polys == None):
                raise Exception('No Features specified')
                return

            ##extract feature points
            if (feature_points != None):
                data_feature_points=dict()
                for name in class_names:
                    vtklist=dict_vtkfiles[name]
                    dataset, extracted_feature_points_info=self.load_features_class(vtklist, min_num_shapes=min_num_shapes_per_class, feature_points = feature_points)
                    data_feature_points[name]=dataset                

            ##extract feature polys
            if (feature_polys != None):
                data_feature_polys=dict()
                for name in class_names:
                    vtklist=dict_vtkfiles[name]
                    dataset, extracted_feature_polys_info=self.load_features_class(vtklist, min_num_shapes=min_num_shapes_per_class, feature_polys = feature_polys)
                    data_feature_polys[name]=dataset

            ##if necessary, concatenate data
            if (feature_polys != None and feature_points != None):
                data_feature=dict()
                for name in class_names:
                    data_feature[name]=np.concatenate((data_feature_points[name],data_feature_polys[name]),axis=1)
            elif (feature_polys != None):
                data_feature=data_feature_polys
            elif (feature_points != None):
                data_feature=data_feature_points
            else:
                raise Exception("Unexpected error")
                return


            #generate labels for each class
            data_labels=dict()
            for name in class_names:
                label = class_corres[name]
                shape_number=data_feature[name].shape[0]

                label_array=np.zeros((shape_number,))
                label_array+=label
                data_labels[name]=label_array

            #write TFRecords
            start_id=0
            dict_tfr=dict()
            for name in class_names:
                
                dict_tfr[name]=self.writeRecords(records_path,data_feature[name],data_labels[name],start_id=start_id)
                start_id += len(dict_tfr[name])

            #add tfrecord path to the input csv file, save it in records_path
            output_csv=os.path.join(records_path,os.path.basename(csvFile))
            with open(csvFile,'r') as csvinput:
                with open(output_csv, 'w') as csvoutput:
                    writer = csv.writer(csvoutput, lineterminator='\n')
                    reader = csv.reader(csvinput)

                    all = []
                    row = next(reader)
                    row.append('TFRecord Files')
                    all.append(row)

                    vtk_file_index=row.index('VTK Files')
                    class_index=row.index('Group')
                    for row in reader:
                        vtk_file=row[vtk_file_index]
                        group=row[class_index]
                        tfrecord_index=dict_vtkfiles[group].index(vtk_file)
                        row.append(dict_tfr[group][tfrecord_index])
                        all.append(row)

                    writer.writerows(all)

            #create dataset description
            totalnumber=0
            dict_example_number=dict()
            for name in class_names:
                dict_example_number[name]=len(dict_vtkfiles[name])
                totalnumber+=len(dict_vtkfiles[name])

            dict_dataset_description=dict()
            dict_dataset_description['dataset_type']=dataset_type
            dict_dataset_description['files_description']=output_csv
            dict_dataset_description['original_files_description']=csvFile
            dict_dataset_description['class_names']=class_names
            dict_dataset_description['class_correspondence']=class_corres
            dict_dataset_description['examples_number']=totalnumber
            dict_dataset_description['examples_per_class']=dict_example_number
            try:
                dict_dataset_description['extracted_feature_points_info']=extracted_feature_points_info
            except:
                dict_dataset_description['extracted_feature_points_info']= None
            try:
                dict_dataset_description['extracted_feature_polys_info']=extracted_feature_polys_info
            except:
                dict_dataset_description['extracted_feature_polys_info']=None

            description_path=os.path.join(records_path,'dataset_description.json')
            try:
              with open(description_path, 'w') as f:
                  json.dump(dict_dataset_description, f,indent = 4)
            except Exception as e:
              print('Unable to save extraction description to', description_path, ':', e)

            return description_path



        elif dataset_type=='generation':
            #extract features for each shape
            ##check for existing features
            if (feature_points == None and feature_polys == None):
                raise Exception('No features to extract from input shape')
                return

            if (feature_points_output == None and feature_polys_output == None):
                raise Exception('No features to extract from output shape')
                return

            # dict_vtkfiles_in
            # dict_vtkfiles_out

            ##extract input feature points
            if (feature_points != None):
                input_data_feature_points=[]
                for file_name in vtkfiles_in:
                    feature, extracted_input_feature_points_info=self.load_features(file_name, feature_points = feature_points)
                    input_data_feature_points.append(feature)              

            ##extract input feature polys
            if (feature_polys != None):
                input_data_feature_polys=[]
                for file_name in vtkfiles_in:
                    feature, extracted_input_feature_polys_info=self.load_features(file_name, feature_polys = feature_polys)
                    input_data_feature_polys.append(feature)

            ##extract output feature points
            if (feature_points_output != None):
                output_data_feature_points=[]
                for file_name in vtkfiles_out:
                    feature, extracted_output_feature_points_info=self.load_features(file_name, feature_points = feature_points_output)
                    output_data_feature_points.append(feature)              

            ##extract output feature polys
            if (feature_polys_output != None):
                output_data_feature_polys=[]
                for file_name in vtkfiles_out:
                    feature, extracted_output_feature_polys_info=self.load_features(file_name, feature_polys = feature_polys_output)
                    output_data_feature_polys.append(feature)

            ##if necessary, concatenate data
            ##input
            if (feature_polys != None and feature_points != None):
                input_data_feature=[]
                for i in range(len(input_data_feature_polys)):
                    input_data_feature.append(np.concatenate((input_data_feature_points[i],input_data_feature_polys[i]),axis=0))
            elif (feature_polys != None):
                input_data_feature=input_data_feature_polys
            elif (feature_points != None):
                input_data_feature=input_data_feature_points
            else:
                raise Exception("Unexpected error")
                return
            ##output
            if (feature_polys_output != None and feature_points_output != None):
                output_data_feature=[]
                for i in range(len(input_data_feature_polys)):
                    output_data_feature.append(np.concatenate((output_data_feature_points[i],output_data_feature_polys[i]),axis=0))
            elif (feature_polys_output != None):
                output_data_feature=output_data_feature_polys
            elif (feature_points_output != None):
                output_data_feature=output_data_feature_points
            else:
                raise Exception("Unexpected error")
                return


            input_data_feature=np.array(input_data_feature)
            output_data_feature=np.array(output_data_feature)

            #write TFRecords

            
            tfr_list=self.writeRecords(records_path,input_data_feature,output_data_feature,start_id=0)

            #add tfrecord path to the input csv file, save it in records_path
            output_csv=os.path.join(records_path,os.path.basename(csvFile))
            with open(csvFile,'r') as csvinput:
                with open(output_csv, 'w') as csvoutput:
                    writer = csv.writer(csvoutput, lineterminator='\n')
                    reader = csv.reader(csvinput)

                    all = []
                    row = next(reader)
                    row.append('TFRecord Files')
                    all.append(row)

                    input_file=row.index('Input VTK Files')
                    for row in reader:
                        vtk_file=row[input_file]
                        tfrecord_index=vtkfiles_in.index(vtk_file)
                        row.append(tfr_list[tfrecord_index])
                        all.append(row)

                    writer.writerows(all)

            #create dataset description

            dict_dataset_description=dict()
            dict_dataset_description['dataset_type']=dataset_type
            dict_dataset_description['files_description']=output_csv
            dict_dataset_description['original_files_description']=csvFile
            dict_dataset_description['examples_number']=len(vtkfiles_in)
            try:
                dict_dataset_description['extracted_input_feature_points_info']=extracted_input_feature_points_info
            except:
                dict_dataset_description['extracted_input_feature_points_info']= None
            try:
                dict_dataset_description['extracted_input_feature_polys_info']=extracted_input_feature_polys_info
            except:
                dict_dataset_description['extracted_input_feature_polys_info']=None
            try:
                dict_dataset_description['extracted_output_feature_points_info']=extracted_output_feature_points_info
            except:
                dict_dataset_description['extracted_output_feature_points_info']= None
            try:
                dict_dataset_description['extracted_output_feature_polys_info']=extracted_output_feature_polys_info
            except:
                dict_dataset_description['extracted_output_feature_polys_info']=None



            description_path=os.path.join(records_path,'dataset_description.json')
            try:
              with open(description_path, 'w') as f:
                  json.dump(dict_dataset_description, f,indent = 4)
            except Exception as e:
              print('Unable to save extraction description to', description_path, ':', e)

            return description_path





        return 'humm la bonne description'

