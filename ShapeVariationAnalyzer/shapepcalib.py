from __future__ import print_function
from __future__ import division
from sklearn.decomposition import PCA
from scipy import stats
from copy import deepcopy
from cpns.cpns import CPNS
from vtk.util.numpy_support import vtk_to_numpy

import numpy as np
import json
import os
import pickle
import sys
import vtk


class BoundError(Exception):
    def __init__(self, value):
        self.value = value


class CSVFileError(Exception):
    def __init__(self,value):
        self.value=value


class JSONFileError(Exception):
    def __init__(self,value):
        self.value=value


class pcaExplorer(object):

    def __init__(self):
        self.dict_data=None
        self.dictPCA={}

        self.useHiddenEigenmodes = True
        self.visibleEigenmodes = 8

        self.isSRep = False

    def useHiddenModes(self,bl):
        self.useHiddenEigenmodes = bl

    def setNumberOfVisibleEigenmodes(self,nb):
        self.visibleEigenmodes = nb
    #basic function
    def loadCSVFile(self,file_path):
        # Check if it's a CSV file
        condition1 = self.checkExtension(file_path, ".csv")
        if not condition1:
            raise CSVFileError('File should be a CSV file')
            return

        # Download the CSV file
        self.original_files=file_path
        self.table = self.readCSVFile(file_path)
        self.dictVTKFiles = dict()
        condition1 = self.creationDictVTKFiles(self.dictVTKFiles)
        condition2 = self.checkSeveralMeshInDict(self.dictVTKFiles)
        #condition3 = self.logic.checkOneMeshPerGroupInDict(self.dictVTKFiles)

        # If the file is not conformed:
        #    Re-initialization of the dictionary containing all the data
        #    which will be used to create a new Classification Groups
        if (not condition2) or (not condition1) :
            self.dictVTKFiles = None
            return

    def process(self):
        #clean PCAdata dict
        self.dictPCA = dict()
        min_explained=0

        all_data=None
        all_files=[]
        group_names = []
        group_data = []
        group_files = []
        group_keys = []
        self.polydata = None

        # For each group, record data and group name
        for key, value in self.dictVTKFiles.items():
            #read data of the group
            data, polydata, group_name = self.readPCAData(value)  # if S-Rep, return data and polydata as None
            group_names.append(group_name)
            group_data.append(data)
            group_files.append(value)
            group_keys.append(key)
            all_files.extend(value)

            if self.polydata is None:
                self.polydata = polydata

        # Compute PCA for each group
        if self.isSRep is True:  # S-Rep
            self.CPNSModel = CPNS()
            self.CPNSModel.setInputFileList(all_files)
            self.CPNSModel.Update()
            all_data = np.transpose(self.CPNSModel.getZCompMatrix())
            self.polydata = self.CPNSModel.getPolyData(np.zeros((all_data.shape[1], 1)))

            num_samples = 0
            for i in range(len(group_files)):
                num_samples_new = num_samples + len(group_files[i])
                data = all_data[num_samples:num_samples_new, :]
                pca_model = self.processPCA(data, group_names[i], group_files[i])
                self.dictPCA[group_keys[i]] = pca_model
                num_samples = num_samples_new
        else:  # vtkPolyData
            for i in range(len(self.dictVTKFiles)):
                #compute PCA and store model in a dict
                pca_model = self.processPCA(group_data[i], group_names[i], group_files[i])
                self.dictPCA[group_keys[i]] = pca_model

                #store data
                if all_data is None:
                    all_data = deepcopy(group_data[i])
                else:
                    all_data = np.concatenate((all_data, group_data[i]), axis=0)

        #compute PCA for all the data
        pca_model = self.processPCA(all_data, "All", all_files)
        self.dictPCA["All"] = pca_model

        self.polydataMean=vtk.vtkPolyData()
        self.polydataMean.DeepCopy(self.polydata)
        self.polydataExploration=vtk.vtkPolyData()
        self.polydataExploration.DeepCopy(self.polydata)

        self.initExploration()

    def save(self,JSONpath):
        
        directory = os.path.dirname(JSONpath)
        basename = os.path.basename(JSONpath)
        name,ext=os.path.splitext(basename)

        self.PYCpath=os.path.join(directory,name+".pyc")

        json_dict,pickle_dict,polydata_dict=self.extractData()

        for ID,polydata in polydata_dict.items():
            vtkfilepath=os.path.join(directory,name+'_'+str(ID)+'_mean.vtk')
            self.saveVTKFile(polydata,vtkfilepath)
            json_dict[ID]["mean_file_path"]=vtkfilepath
            print("VTK File: " + vtkfilepath)
            sys.stdout.flush()

        json_dict["original_files"] = self.original_files
        json_dict["python_objects_path"] = self.PYCpath
        json_dict["is_srep"] = self.isSRep

        with open(JSONpath,'w') as jsonfile:
            json.dump(json_dict,jsonfile,indent=4)
        print("JSON File: " + JSONpath)
        sys.stdout.flush()

        with open(self.PYCpath,'wb') as pycfile:
            pickle.dump(pickle_dict,pycfile)
        print("PYC File: " + self.PYCpath)
        sys.stdout.flush()

        print("Files Saved")
        sys.stdout.flush()

    def load(self,JSONfile):

        self.dictVTKFiles=None
        condition1 = self.checkExtension(JSONfile, ".json")
        if not condition1:
            raise JSONFileError('File should be a JSON file')
            return

        with open(JSONfile,'r') as jsonfile:
            json_dict = json.load(jsonfile)

        self.PYCpath=json_dict["python_objects_path"]

        with open(self.PYCpath, 'rb') as pycfile:
            pickle_dict = pickle.load(pycfile)



        self.loadExploration(json_dict,pickle_dict)


        self.initExploration()

    def setCurrentPCAModel(self, keygroup):

        self.current_pca_model=self.dictPCA[keygroup]
        self.current_group_key=keygroup

        self.initPolyDataExploration()
        self.initPolyDataMean()

    #Model Evaluation
    def computeCompactness(self,M_max=8,group=None):
        '''if 'compactness' in self.current_pca_model.keys():
            return'''
        if group:
            current_model=self.dictPCA[group]
        else:
            current_model=self.current_pca_model
        eig=current_model['eigenvalues']
        n=current_model['num_components']

        compactness=list()
        compactness_ste=list()

        for M in range(1,M_max+1):
            compactness.append(np.sum(eig[:M]))
            compactness_ste.append(np.sqrt(2.0/n)*np.sum(eig[:M]))

        compactness=np.array(compactness)
        compactness_ste=np.array(compactness_ste)

        current_model['compactness']=compactness
        current_model['compactness_ste']=compactness_ste

        return compactness,compactness_ste

    def computeSpecificity(self,M_max=8,shape_number=10000,group=None):

        if group:
            current_model=self.dictPCA[group]
        else:
            current_model=self.current_pca_model

        #get the pca object
        pca_model=current_model['pca']

        #restore data
        loads=current_model["data_projection"]
        data=pca_model.inverse_transform(loads)


        #Getting load means and std
        loads_mean=current_model["data_projection_mean"]
        loads_std=current_model["data_projection_std"]


        #get the number of component of the pca model
        n_component=current_model["num_components"]

        #init specificity and specificity_std lists
        specificity=list()
        specificity_std=list()



        for M in range(1,M_max+1):
            #Loads generation
            means=np.append(loads_mean[:M],np.zeros(n_component-M))
            stds=np.append(loads_std[:M],np.zeros(n_component-M))

            random_loads=np.random.normal(means,stds,(shape_number,n_component))

            #for each random shape, find the nearest shape
            distance_list=list()
            for i in range(shape_number):
                rload=random_loads[i,:]
                rshape=pca_model.inverse_transform(rload)

                dist=self.findMinimalDistance(rshape,data)
                distance_list.append(dist)

            #compute specificity and specificity_std at M
            specificity.append(np.mean(distance_list))
            specificity_std.append(np.std(distance_list)/np.sqrt(shape_number))

        #conversion into a numpy array
        specificity=np.array(specificity)
        specificity_std=np.array(specificity_std)

        #store the data
        current_model['specificity']=specificity
        current_model['specificity_std']=specificity_std

        return specificity,specificity_std

    def computeGeneralization(self,M_max=8,group=None):

        if group:
            current_model=self.dictPCA[group]
        else:
            current_model=self.current_pca_model
        #get the pca object
        pca_model=current_model['pca']

        #restore data
        loads=current_model["data_projection"]
        data=pca_model.inverse_transform(loads)

        member_number=data.shape[0]
        n_component=current_model["num_components"]
        dist_list=list()
        for i in range(member_number):


            reduced_data, excluded_member=data[np.arange(member_number)!=i,:],data[i,:]
            reduced_data_mean = np.mean(reduced_data, axis=0, keepdims=True)

            pca_model= PCA()
            pca_model.fit(reduced_data-reduced_data_mean)

            load=pca_model.transform(excluded_member-reduced_data_mean)
            n_component=load.shape[1]
            M_list=list()
            reduced_load=load
            for M in range(1,M_max+1):
                #load[0]=load[0,:M]
                reduced_load=np.append(load[0,:M],np.zeros(n_component-M))
                reduced_load=np.array([reduced_load])
                reconstruction =pca_model.inverse_transform(reduced_load)+reduced_data_mean
                M_list.append(self.computeDistance(excluded_member,reconstruction))

            dist_list.append(M_list)

        generalization=np.mean(dist_list,axis=0)
        generalization_error=np.std(dist_list,axis=0)/np.sqrt(member_number-1)

        generalization=np.array(generalization)
        generalization_error=np.array(generalization_error)

        current_model['generalization']=generalization
        current_model['generalization_error']=generalization_error

        return generalization,generalization_error

    def evaluationExist(self):
        if 'compactness' in self.current_pca_model.keys():
            return True
        return False

    def findMinimalDistance(self,shape,data):
        min_dist=None
        for i in range(data.shape[0]):
            data_shape=data[i,:]
            shape=shape.reshape(1,-1)
            data_shape=data_shape.reshape(1,-1)
            dist = self.computeDistance(shape,data_shape)
            if min_dist == None:
                min_dist=dist
            elif dist < min_dist:
                min_dist=dist

        return min_dist

    def computeDistance(self,shape1,shape2):
        dist = shape1-shape2
        dist = dist.reshape(-1,3)
        dist = np.linalg.norm(dist,axis=1)
        dist = np.mean(dist)

        return dist

    def extractEvaluationData(self):
        if not self.evaluationExist():
            print("Error no evaluation data found")
            return None

        json_dict={}

        for ID , model in self.dictPCA.items():
            #extract PCA information
            json_dict[ID]={}

            for key , value in model.items():
                if key == 'generalization':
                    json_dict[ID][key]=value.tolist()
                if key == 'generalization_error':
                    json_dict[ID][key]=value.tolist()

                if key == 'specificity':
                    json_dict[ID][key]=value.tolist()
                if key == 'specificity_std':
                    json_dict[ID][key]=value.tolist()

                if key == 'compactness':
                    json_dict[ID][key]=value.tolist()
                if key == 'compactness_ste':
                    json_dict[ID][key]=value.tolist()
        return json_dict



    def saveEvaluation(self,JSONpath):
        condition1 = self.checkExtension(JSONpath, ".json")
        if not condition1:
            raise JSONFileError('File should be a JSON file')
            return



        eval_data=self.extractEvaluationData()

        if eval_data:


            with open(JSONpath,'w') as jsonfile:
                json.dump(eval_data,jsonfile,indent=4)
            print("Evaluation JSON File: " + JSONpath)
            sys.stdout.flush()

        else:
            print("no evaluation found")
            return

    def importEvaluation(self,JSONpath):
        condition1 = self.checkExtension(JSONpath, ".json")
        if not condition1:
            raise JSONFileError('File should be a JSON file')
            return

        with open(JSONpath,'r') as jsonfile:
            json_dict = json.load(jsonfile)


        for grp, val in self.dictPCA.items():
            for key,value in json_dict[str(grp)].items():
                self.dictPCA[grp][key]=np.array(value)

    def updateJSONFile(self,JSONpath):
        condition1 = self.checkExtension(JSONpath, ".json")
        if not condition1:
            raise JSONFileError('File should be a JSON file')
            return

        json_dict,_,_=self.extractData()
        json_dict["original_files"] = self.original_files
        json_dict["python_objects_path"] = self.PYCpath


        eval_data=self.extractEvaluationData()

        if eval_data:
            for group, model in eval_data.items():
                for key, evaluation in eval_data[group].items():
                    json_dict[group][key]=evaluation

            with open(JSONpath,'w') as jsonfile:
                json.dump(json_dict,jsonfile,indent=4)
            print("JSON File: " + JSONpath)
            sys.stdout.flush()

        else:
            print("no evaluation found")
            return

    def reloadJSONFile(self,JSONpath):
        condition1 = self.checkExtension(JSONpath, ".json")
        if not condition1:
            raise JSONFileError('File should be a JSON file')
            return

        with open(JSONpath,'r') as jsonfile:
            json_dict = json.load(jsonfile)


        for ID , model in self.dictPCA.items():
            if ID != "original_files" and ID !="python_objects_path" :
                for key , value in json_dict[str(ID)].items():
                    if type(value)==type(list()):
                        self.dictPCA[ID][key]=np.array(value)
                    else:
                        self.dictPCA[ID][key]=value

        return


    #polydata

    def updatePolyDataExploration(self,num_slider,ratio):


        if ratio >=1 or ratio <=-1:
            raise BoundError('ratio should be between -1 and 1, value :'+str(ratio))


        #update current loads
        pca_mean=self.current_pca_model["data_projection_mean"]
        pca_std=self.current_pca_model["data_projection_std"]

        PCA_model=self.current_pca_model['pca']
        PCA_current_loads = self.current_pca_model["current_pca_loads"]

        mean =self.current_pca_model['data_mean']

        X=1-(((ratio)+1)/2.0)


        PCA_current_loads[num_slider]=pca_mean[num_slider]+stats.norm.isf(X)*pca_std[num_slider]
        #print(self.PCA_current_loads)
        sys.stdout.flush()

        if self.useHiddenEigenmodes == True:
            self.pca_points_numpy=PCA_model.inverse_transform(PCA_current_loads)+mean
        else:
            numberofmodes=len(PCA_current_loads)
            loads=np.concatenate((PCA_current_loads[:self.visibleEigenmodes],np.zeros(numberofmodes-self.visibleEigenmodes)))
            self.pca_points_numpy=PCA_model.inverse_transform(loads)+mean


        self.modifyVTKPointsFromNumpy(self.pca_points_numpy[0])
        self.generateDistanceColor()
        self.autoOrientNormals(self.polydataExploration)
        self.polydataExploration.Modified()

        #print(pca_points.reshape(1002,3))
        #sys.stdout.flush()
        #return  self.current_pca_model["polydata"]

    def setCurrentShapeFromId(self,Id):
        population_projection=self.current_pca_model["data_projection"]
        #if Id == 141:
         #   print(population_projection[Id,:])
        self.current_pca_model["current_pca_loads"] = deepcopy(population_projection[Id,:])

        PCA_model=self.current_pca_model['pca']
        PCA_current_loads=self.current_pca_model["current_pca_loads"]
        mean =self.current_pca_model['data_mean']

        if self.useHiddenEigenmodes == True:
            self.pca_points_numpy=PCA_model.inverse_transform(PCA_current_loads)+mean
        else:
            numberofmodes=len(PCA_current_loads)
            loads=np.concatenate((PCA_current_loads[:self.visibleEigenmodes],np.zeros(numberofmodes-self.visibleEigenmodes)))
            self.pca_points_numpy=PCA_model.inverse_transform(loads)+mean


        self.modifyVTKPointsFromNumpy(self.pca_points_numpy[0])
        self.generateDistanceColor()

        self.autoOrientNormals(self.polydataExploration)
        #self.polydataExploration.Modified()

    def setCurrentShapeFromIdList(self,Idlist):
        population_projection=self.current_pca_model["data_projection"]
        #if Id == 141:
         #   print(population_projection[Id,:])
        loadslist= deepcopy(population_projection[Idlist,:])

        self.current_pca_model["current_pca_loads"] = np.mean(loadslist, axis=0, keepdims=True)[0]


        PCA_model=self.current_pca_model['pca']
        PCA_current_loads=self.current_pca_model["current_pca_loads"]
        mean =self.current_pca_model['data_mean']

        if self.useHiddenEigenmodes == True:
            self.pca_points_numpy=PCA_model.inverse_transform(PCA_current_loads)+mean
        else:
            numberofmodes=len(PCA_current_loads)
            loads=np.concatenate((PCA_current_loads[:self.visibleEigenmodes],np.zeros(numberofmodes-self.visibleEigenmodes)))
            self.pca_points_numpy=PCA_model.inverse_transform(loads)+mean


        self.modifyVTKPointsFromNumpy(self.pca_points_numpy[0])
        self.generateDistanceColor()

        self.autoOrientNormals(self.polydataExploration)
        self.polydataExploration.Modified()

    def resetPCAPolyData(self):
        num_components=self.current_pca_model["num_components"]
        self.current_pca_model["current_pca_loads"] = np.zeros(num_components)
        self.pca_points_numpy=self.current_pca_model['data_mean']

        self.modifyVTKPointsFromNumpy(self.pca_points_numpy[0])
        self.autoOrientNormals(self.polydataExploration)
        self.polydataExploration.Modified()

    def getPolyDataExploration(self):

        return self.polydataExploration

    def getPolyDataMean(self):

        return self.polydataMean

    #Plots
    def getPCAProjections(self, normalized=False):
        X_pca = self.current_pca_model["data_projection"]
        X_std = self.current_pca_model["data_projection_std"]

        pc1 = X_pca[:,0].flatten()
        if normalized is True:
            pc1 = pc1 / X_std[0]
        pc1 = self.generateVTKFloatFromNumpy(pc1)
        #pc1 = numpy_to_vtk(num_array=pc1, array_type=vtk.VTK_FLOAT)


        pc2 = X_pca[:,1].flatten()
        if normalized is True:
            pc2 = pc2 / X_std[1]
        pc2 = self.generateVTKFloatFromNumpy(pc2)
        #pc2 = numpy_to_vtk(num_array=pc2, array_type=vtk.VTK_FLOAT)

        return pc1, pc2

    def getPCAProjectionLabels(self):
        labels = self.current_pca_model["source_files"]
        vtkLabels = vtk.vtkStringArray()
        for name in labels:
            path, file = os.path.split(name)
            group = os.path.basename(path)
            vtkLabels.InsertNextValue(" - " + group + "/" + file)
        return vtkLabels

    def getPlotLevel(self,num_component):

        level95=np.ones(num_component)*95
        level1=np.ones(num_component)
        #xlevel=vtk.util.numpy_support.numpy_to_vtk(num_array=xlevel, array_type=vtk.VTK_FLOAT)
        level95= self.generateVTKFloatFromNumpy(level95)
        level1= self.generateVTKFloatFromNumpy(level1)
        #level95=numpy_to_vtk(num_array=level95, array_type=vtk.VTK_FLOAT)
        #level1=numpy_to_vtk(num_array=level1, array_type=vtk.VTK_FLOAT)

        return  level95, level1

    def getPCAVarianceExplainedRatio(self,num_component):
        evr = self.current_pca_model['explained_variance_ratio'][0:num_component].flatten()*100
        sumevr = np.cumsum(evr)
        evr = self.generateVTKFloatFromNumpy(evr)
        sumevr= self.generateVTKFloatFromNumpy(sumevr)

        x = np.arange(1,num_component+1).flatten()
        x = self.generateVTKFloatFromNumpy(x)
        return x,evr,sumevr

    def getCompactness(self):
        compac = self.current_pca_model['compactness']
        compac_err = self.current_pca_model['compactness_ste']
        x = np.arange(1,compac.shape[0]+1).flatten()

        compac = self.generateVTKFloatFromNumpy(compac)
        compac_err = self.generateVTKFloatFromNumpy(compac_err)
        x = self.generateVTKFloatFromNumpy(x)

        return x,compac,compac_err

    def getSpecificity(self):
        spec = self.current_pca_model['specificity']
        spec_err = self.current_pca_model['specificity_std']
        x = np.arange(1,spec.shape[0]+1).flatten()

        spec = self.generateVTKFloatFromNumpy(spec)
        spec_err = self.generateVTKFloatFromNumpy(spec_err)
        x = self.generateVTKFloatFromNumpy(x)

        return x,spec,spec_err

    def getGeneralization(self):
        gene = self.current_pca_model['generalization']
        gene_err = self.current_pca_model['generalization_error']
        x = np.arange(1,gene.shape[0]+1).flatten()

        gene = self.generateVTKFloatFromNumpy(gene)
        gene_err = self.generateVTKFloatFromNumpy(gene_err)
        x = self.generateVTKFloatFromNumpy(x)

        return x,gene,gene_err

    #gets
    def getCurrentRatio(self,num_slider):
        pca_mean=self.current_pca_model["data_projection_mean"][num_slider]
        pca_std=self.current_pca_model["data_projection_std"][num_slider]

        PCA_current_loads = self.current_pca_model["current_pca_loads"] [num_slider]

        '''X=1-(((ratio/1000.0)+1)/2.0)
        stats.norm.isf(X)
        PCA_current_loads[num_slider]=pca_mean[num_slider]+stats.norm.isf(X)*pca_std[num_slider]'''

        '''if num_slider==4:
            print((PCA_current_loads-pca_mean)/(pca_std))'''

        ratio = (PCA_current_loads-pca_mean)/(pca_std)
        ratio = stats.norm.sf(ratio)
        ratio = 1000*((2*(1-ratio))-1)
        return int(ratio)

    def getNumComponent(self):

        return self.current_pca_model["num_components"]

    def getRelativeNumComponent(self,min_explained):
        explained_variance_ratio=self.current_pca_model['explained_variance_ratio']
        num_components = 0

        for evr in explained_variance_ratio:
            #print(num_components+1,evr)
            if evr < min_explained:
                break
            if evr < 1e-12:
                print('Component %d ignored because it is not revelant (explained variance ratio < 1e-12)'%(num_components+1) )
            else:
                num_components += 1
        return num_components

    def getExplainedRatio(self):

        return self.current_pca_model["explained_variance_ratio"]

    def getDataStd(self):

        return self.current_pca_model["data_std"]

    def getColor(self):
        (r,g,b)=self.current_pca_model['color']
        return r,g,b

    def getColorParam(self):

        return self.colormodeparam1,self.colormodeparam2

    def getDictPCA(self):

        return self.dictPCA

    def getGroups(self):

        return list(self.dictPCA.keys())

    #sets
    def changeCurrentGroupColor(self,color):

        self.current_pca_model['color']=color

    def setColorModeParam(self,param1,param2):
        self.colormodeparam1=param1
        self.colormodeparam2=param2

    def setColorMode(self,colormode):
        self.colormode = colormode

        '''if colormode == 1: #unsigned distance
            explorationnode=slicer.mrmlScene.GetFirstNodeByName('PCA Exploration')
            colornode = slicer.mrmlScene.GetFirstNodeByName('PCA Unsigned Distance Color Table')
            if (explorationnode is not None) and (colornode is not None):
                explorationnode.GetDisplayNode().SetAndObserveColorNodeID(colornode.GetID())
                #explorationnode.SetInterpolate(1)
                explorationnode.Modified()

        if colormode == 2: #signed distance
            explorationnode=slicer.mrmlScene.GetFirstNodeByName('PCA Exploration')
            colornode = slicer.mrmlScene.GetFirstNodeByName('PCA Signed Distance Color Table')
            if (explorationnode is not None) and (colornode is not None):
                explorationnode.GetDisplayNode().SetAndObserveColorNodeID(colornode.GetID())
                #explorationnode.SetInterpolate(1)
                explorationnode.Modified()'''

        self.generateDistanceColor()


    #PCA
    def readPCAData(self, fileList):
        """
        Read data from fileList and format it for PCA computation
        """

        # get polydata statistics

        y_design = []
        numpoints = -1
        nshape = 0
        polydata = 0
        group_name=None

        for vtkfile in fileList:
            if vtkfile.endswith(".vtk"):
                #print("Reading", vtkfile)
                reader = vtk.vtkPolyDataReader()
                reader.SetFileName(vtkfile)
                reader.Update()
                shapedata = reader.GetOutput()
                #self.polyDataPCA=shapedata
                shapedatapoints = shapedata.GetPoints()


                if polydata == 0:
                    polydata = shapedata
                if group_name is None:
                    group_name = os.path.basename(os.path.dirname(vtkfile))

                y_design.append([])

                if numpoints == -1:
                    numpoints = shapedatapoints.GetNumberOfPoints()

                if numpoints != shapedatapoints.GetNumberOfPoints():
                    print("WARNING! File ignored, the number of points is not the same for the shape:", vtkfile)
                    sys.stdout.flush()
                    pass

                for i in range(shapedatapoints.GetNumberOfPoints()):
                    p = shapedatapoints.GetPoint(i)
                    y_design[nshape].append(p)
                nshape+=1
            elif vtkfile.endswith(".xml"):
                self.isSRep = True
                if group_name is None:
                    group_name = os.path.basename(os.path.dirname(vtkfile))

        if (self.isSRep is True):
            if (len(y_design) is 0):
                print("Reading s-rep dataset")
                y_design = None
                polydata = None
            else:
                print("Error: input folder contains both .vtk and srep(.xml) files!")
                y_design = np.array(y_design)
                y_design = y_design.reshape(y_design.shape[0], -1)
        else:
            if len(y_design) is not 0:
                print("Reading poly dataset")
                y_design = np.array(y_design)
                y_design = y_design.reshape(y_design.shape[0], -1)
            else:
                print("No input from group: ", group_name)
        return y_design, polydata, group_name


    def processPCA(self,X,group_name, fileList):
        X_ = np.mean(X, axis=0, keepdims=True)
        X_std = np.std(X,axis=0,keepdims=True)


        pca = PCA()

        X_pca=pca.fit_transform(X - X_)

        X_pca_mean = np.mean(X_pca, axis=0, keepdims=True)
        X_pca_std = np.std(X_pca, axis=0, keepdims=True)

        pca_model = {}
        pca_model["pca"] = pca
        pca_model['explained_variance_ratio']=pca.explained_variance_ratio_
        pca_model["eigenvalues"]=np.multiply(pca.singular_values_,pca.singular_values_)
        pca_model["components"]=pca.components_
        pca_model["num_components"]=pca.components_.shape[0]
        pca_model["data_mean"] = X_
        pca_model["data_std"] = X_std
        pca_model["data_projection"]=X_pca
        pca_model["data_projection_mean"]=X_pca_mean[0]
        pca_model["data_projection_std"]=X_pca_std[0]
        pca_model["current_pca_loads"] = np.zeros(pca.components_.shape[0])
        pca_model["group_name"]=group_name

        #generate a random color
        color = list(np.random.choice(range(256), size=3) / 255.0)
        pca_model["color"]=color
        pca_model["source_files"]=fileList

        return pca_model
    def extractData(self):
        json_dict={}
        pickle_dict={}
        polydata_dict={}


        for ID , model in self.dictPCA.items():
            #extract mean shape for the ID group
            polydata_dict[ID]=self.generateMeanShape(ID)
            #extract PCA information
            json_dict[ID]={}
            pickle_dict[ID]={}
            for key , value in model.items():
                if type(value).__module__ == '__builtin__':
                    json_dict[ID][key]=value
                elif type(value).__module__ == np.__name__:
                    json_dict[ID][key]=value.tolist()
                else:
                    pickle_dict[ID][key]=value


        return json_dict,pickle_dict,polydata_dict
    def loadExploration(self,json_dict,pickle_dict):
        self.dictPCA={}

        for ID , model in pickle_dict.items():
            if ID != "original_files" and ID !="python_objects_path" :
                self.dictPCA[ID]={}
                for key , value in model.items():
                    self.dictPCA[ID][key]=value
                for key , value in json_dict[str(ID)].items():
                    if type(value)==type(list()):
                        self.dictPCA[ID][key]=np.array(value)
                    else:
                        self.dictPCA[ID][key]=value


        polydata=self.readVTKfile(json_dict["0"]["mean_file_path"])
        self.polydata=vtk.vtkPolyData()
        self.polydata.DeepCopy(polydata)


        self.polydataMean=vtk.vtkPolyData()
        self.polydataMean.DeepCopy(polydata)
        self.polydataExploration=vtk.vtkPolyData()
        self.polydataExploration.DeepCopy(polydata)

        self.original_files=json_dict["original_files"]
        #self.dictPCA["original_files"] = json_dict["original_files"]
        if json_dict["is_srep"] is True:
            self.isSRep = True
            self.table = self.readCSVFile(self.original_files)
            all_files = []
            for i in range(self.table.GetNumberOfRows()):
                all_files.append(self.table.GetValue(i, 0).ToString())
            self.CPNSModel = CPNS()
            self.CPNSModel.setInputFileList(all_files)
            self.CPNSModel.Update()

    #Common
    def initExploration(self):
        self.setColorModeParam(10,10)
        self.setColorMode(0)
        self.setCurrentPCAModel(0)
    def readVTKfile(self,filename):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()
        data = reader.GetOutput()
        return data
    def modifyVTKPointsFromNumpy(self,npArray):
        if self.isSRep is False:
            num_points = int(npArray.shape[0] / 3)
            for i in range(num_points):
                self.pca_exploration_points.SetPoint(i,npArray[3*i],npArray[3*i+1],npArray[3*i+2])
        else:
            new_points = self.CPNSModel.getPolyData(npArray[:, np.newaxis]).GetPoints()
            self.pca_exploration_points.DeepCopy(new_points)
        self.pca_exploration_points.Modified()
    def generateVTKPointsFromNumpy(self,npArray):
        if self.isSRep is False:
            num_points = int(npArray.shape[0]/3)
            vtk_points = vtk.vtkPoints()
            for i in range(num_points):
                vtk_points.InsertNextPoint(npArray[3*i],npArray[3*i+1],npArray[3*i+2])
        else:
            vtk_points = self.CPNSModel.getPolyData(npArray[:, np.newaxis]).GetPoints()
        return vtk_points
    def generateVTKFloatFromNumpy(self,np_array):
        size = np_array.size

        vtk_float = vtk.vtkFloatArray()
        vtk_float.SetNumberOfComponents(1)
        for i in range(size):
            vtk_float.InsertNextTuple([np_array[i]])
        return vtk_float
    def checkExtension(self, filename, extension):
        """ Check if the path given has the right extension
        """
        if os.path.splitext(os.path.basename(filename))[1] == extension :
            return True
        elif os.path.basename(filename) == "" or os.path.basename(filename) == " " :
            return False
        print('Wrong extension file, a ' + extension + ' file is needed!')
        return False
    def readCSVFile(self, filename):
        """ Function to read a CSV file
        """
        print("CSV FilePath: " + filename)
        sys.stdout.flush()
        CSVreader = vtk.vtkDelimitedTextReader()
        CSVreader.SetFieldDelimiterCharacters(",")
        CSVreader.SetFileName(filename)
        CSVreader.SetHaveHeaders(True)
        CSVreader.Update()

        return CSVreader.GetOutput()
    def creationDictVTKFiles(self, dict):
        """ Function to create a dictionary containing all the vtk filepaths sorted by group
            - the paths are given by a CSV file
            - If one paths doesn't exist
                Return False
            Else if all the path of all vtk file exist
            Return True
        """
        for i in range(0, self.table.GetNumberOfRows()):
            if not os.path.exists(self.table.GetValue(i,0).ToString()):
                msg='VTK file not found, path not good at lign ' + str(i+2)
                raise CSVFileError(msg)
                return False
            value = dict.get(self.table.GetValue(i,1).ToInt(), None)
            if value == None:
                dict[self.table.GetValue(i,1).ToInt()] = self.table.GetValue(i,0).ToString()
            else:
                if type(value) is type(list()):
                    value.append(self.table.GetValue(i,0).ToString())
                else:
                    tempList = list()
                    tempList.append(value)
                    tempList.append(self.table.GetValue(i,0).ToString())
                    dict[self.table.GetValue(i,1).ToInt()] = tempList

        return True
    def checkSeveralMeshInDict(self, dict):
        """ Function to check if in each group
        there is at least more than one mesh
        """
        for key, value in dict.items():
            if type(value) is not type(list()) or len(value) == 1:
                msg='The group ' + str(key) + ' must contain more than one mesh.'
                raise CSVFileError(msg)
                return False
        return True
    def saveVTKFile(self, polydata, filepath):
        """ Function to save a VTK file to the filepath given
        """
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(filepath)
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(polydata)
        else:
            writer.SetInputData(polydata)
        writer.Update()
        writer.Write()
        return
    #polydata and color
    def autoOrientNormals(self, model):
        normals = vtk.vtkPolyDataNormals()
        normals.SetAutoOrientNormals(True)
        normals.SetFlipNormals(False)
        normals.SetSplitting(False)
        normals.ConsistencyOn()
        normals.SetInputData(model)
        normals.Update()
        normalspoint=normals.GetOutput().GetPointData().GetArray("Normals")
        model.GetPointData().SetNormals(normalspoint)

        return normals.GetOutput()
    def initPolyDataExploration(self):
        PCA_model=self.current_pca_model['pca']
        PCA_current_loads = self.current_pca_model["current_pca_loads"]
        mean =self.current_pca_model['data_mean']

        if self.useHiddenEigenmodes == True:
            self.pca_points_numpy=PCA_model.inverse_transform(PCA_current_loads)+mean
        else:
            numberofmodes=len(PCA_current_loads)
            loads=np.concatenate((PCA_current_loads[:self.visibleEigenmodes],np.zeros(numberofmodes-self.visibleEigenmodes)))
            self.pca_points_numpy=PCA_model.inverse_transform(loads)+mean

        self.pca_exploration_points = self.generateVTKPointsFromNumpy(self.pca_points_numpy[0])
        self.polydataExploration.SetPoints(self.pca_exploration_points)

        self.autoOrientNormals(self.polydataExploration)
        self.generateDistanceColor()

        self.polydataExploration.Modified()
    def initPolyDataMean(self):
        mean = self.current_pca_model['data_mean']

        mean_points = self.generateVTKPointsFromNumpy(mean[0])

        self.polydataMean.SetPoints(mean_points)

        self.autoOrientNormals(self.polydataMean)

        self.polydataMean.Modified()
    def generateMeanShape(self,ID):
        mean =self.dictPCA[ID]['data_mean']

        mean_mesh=self.generateVTKPointsFromNumpy(mean[0])

        polydata=vtk.vtkPolyData()
        polydata.DeepCopy(self.polydata)
        polydata.SetPoints(mean_mesh)

        self.autoOrientNormals(polydata)

        return polydata
    def generateDistanceColor(self):
        if self.colormode==0:
            return

        if self.isSRep is False:
            mean = self.current_pca_model['data_mean'][0]
            exploration_points=self.pca_points_numpy[0]
        else:
            mean = self.CPNSModel.getPolyData(np.transpose(self.current_pca_model['data_mean']))
            mean = vtk_to_numpy(mean.GetPoints().GetData()).reshape(-1)
            exploration_points = self.CPNSModel.getPolyData(np.transpose(self.pca_points_numpy))
            exploration_points = vtk_to_numpy(exploration_points.GetPoints().GetData()).reshape(-1)

        if self.colormode == 1:
            colors = self.unsignedDistance(mean, exploration_points)

        if self.colormode == 2:
            colors = self.signedDistance(mean, exploration_points)

        self.polydataExploration.GetPointData().SetScalars(colors)
        self.polydataExploration.GetPointData().Modified()
        #self.polydataExploration.Modified()
    def signedDistance(self,mean,exploration_points):
        colors = vtk.vtkFloatArray()
        colors.SetName("Distance")
        colors.SetNumberOfComponents(1)

        max_inside=self.colormodeparam2
        max_outside=self.colormodeparam1

        red = np.array([max_outside])
        white =  np.array([(-max_inside+max_outside)/2.0])
        blue = np.array([-max_inside])
        color=np.array([50])

        select_enclosed_points=vtk.vtkSelectEnclosedPoints()
        select_enclosed_points.SetInputData(self.polydataExploration)
        select_enclosed_points.SetSurfaceData(self.polydataMean)
        select_enclosed_points.SetTolerance(0.000001)
        select_enclosed_points.Update()

        for i in range(0,len(mean),3):
            point=exploration_points[i:i+3]
            meanpoint=mean[i:i+3]
            distance = np.linalg.norm(point-meanpoint)

            if select_enclosed_points.IsInside(int(i/3)) == 1:
                ratio=distance/max_inside
                color=ratio*blue+(1-ratio)*white
            else:
                ratio=distance/max_outside
                color=ratio*red+(1-ratio)*white

            colors.InsertNextTuple(color)

        colors.Modified()
        return colors
    def unsignedDistance(self,mean,exploration_points):
        colors = vtk.vtkFloatArray()
        colors.SetName("Distance")
        colors.SetNumberOfComponents(1)

        color=np.array([0.0])

        for i in range(0,len(mean),3):
            point=exploration_points[i:i+3]
            meanpoint=mean[i:i+3]
            distance = np.linalg.norm(point-meanpoint)

            color[0]=distance

            colors.InsertNextTuple(color)#ratio*blue+(1-ratio)*white)

        colors.Modified()
        return colors

class shapepcalib(object):
    def __init__(self,parent=None):
        #ScriptedLoadableModule.__init__(self, parent)
        parent.title = "shapepcalib"
        parent.categories = ["Shape Analysis.Advanced"]
        parent.dependencies = []
        parent.contributors = ["Lopez Mateo (University of North Carolina)"]
        parent.helpText = """

            """
        parent.acknowledgementText = """

            """