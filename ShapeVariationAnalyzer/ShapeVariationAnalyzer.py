import os, sys
import csv
import unittest
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from types import *
import math
import shutil

import inputData
import pickle
import numpy as np
import zipfile
import json
import subprocess


class ShapeVariationAnalyzer(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "ShapeVariationAnalyzer"
        parent.categories = ["Quantification"]
        parent.dependencies = []
        parent.contributors = ["Priscille de Dumast (University of Michigan), Laura Pascal (University of Michigan)"]
        parent.helpText = """
            Shape Variation Analyzer allows the classification of 3D models, 
            according to their morphological variations. 
            This tool is based on a deep learning neural network.
            """
        parent.acknowledgementText = """
            This work was supported by the National
            Institutes of Dental and Craniofacial Research
            and Biomedical Imaging and Bioengineering of
            the National Institutes of Health under Award
            Number R01DE024450.
            """


class ShapeVariationAnalyzerWidget(ScriptedLoadableModuleWidget):
    def setup(self):

        ScriptedLoadableModuleWidget.setup(self)        

        # ---- Widget Setup ----

        # Global Variables
        self.logic = ShapeVariationAnalyzerLogic(self)
        
        self.dictVTKFiles = dict()
        self.dictGroups = dict()
        self.dictCSVFile = dict()
        self.directoryList = list()
        self.groupSelected = set()
        self.dictShapeModels = dict()
        self.patientList = list()
        self.dictResults = dict()
        self.dictFeatData = dict()

        self.allFeatures = list()
        self.allFeatures.append('Curvedness')
        self.allFeatures.append('Distances to average shapes')
        self.allFeatures.append('Distance to control group')
        self.allFeatures.append('Gaussian Curvature')
        self.allFeatures.append('Maximum Curvature')
        self.allFeatures.append('Minimum Curvature')
        self.allFeatures.append('Mean Curvature')
        self.allFeatures.append('Normals')
        self.allFeatures.append('Position')
        self.allFeatures.append('Shape Index')
        self.featuresList = list()

        # Interface
        self.moduleName = 'ShapeVariationAnalyzer'
        scriptedModulesPath = eval('slicer.modules.%s.path' % self.moduleName.lower())
        scriptedModulesPath = os.path.dirname(scriptedModulesPath)
        path = os.path.join(scriptedModulesPath, 'Resources', 'UI', '%s.ui' % self.moduleName)
        self.widget = slicer.util.loadUI(path)
        self.layout.addWidget(self.widget)

        #     global variables of the Interface:
        #          Tab: Creation of CSV File for Classification Groups
        self.collapsibleButton_creationCSVFile = self.logic.get('CollapsibleButton_creationCSVFile')
        self.spinBox_group = self.logic.get('spinBox_group')
        self.directoryButton_creationCSVFile = self.logic.get('DirectoryButton_creationCSVFile')
        self.stackedWidget_manageGroup = self.logic.get('stackedWidget_manageGroup')
        self.pushButton_addGroup = self.logic.get('pushButton_addGroup')
        self.pushButton_removeGroup = self.logic.get('pushButton_removeGroup')
        self.pushButton_modifyGroup = self.logic.get('pushButton_modifyGroup')
        self.pushButton_exportCSVfile = self.logic.get('pushButton_exportCSVfile')
        #          Tab: Creation of New Classification Groups
        self.collapsibleButton_previewClassificationGroups = self.logic.get('CollapsibleButton_previewClassificationGroups')
        self.pathLineEdit_previewGroups = self.logic.get('pathLineEdit_previewGroups')
        self.collapsibleGroupBox_previewVTKFiles = self.logic.get('CollapsibleGroupBox_previewVTKFiles')
        self.checkableComboBox_ChoiceOfGroup = self.logic.get('CheckableComboBox_ChoiceOfGroup')
        self.tableWidget_VTKFiles = self.logic.get('tableWidget_VTKFiles')
        self.pushButton_previewVTKFiles = self.logic.get('pushButton_previewVTKFiles')

        self.pushButton_exportUpdatedClassification = self.logic.get('pushButton_exportUpdatedClassification')
                 # Tab: Selection Classification Groups
        self.comboBox_healthyGroup = self.logic.get('comboBox_healthyGroup')
        
        #          Tab: Classification Network
        self.collapsibleButton_classificationNetwork = self.logic.get('collapsibleButton_classificationNetwork')
        self.MRMLNodeComboBox_VTKInputData = self.logic.get('MRMLNodeComboBox_VTKInputData')
        self.pathLineEdit_CSVInputData = self.logic.get('PathLineEdit_CSVInputData')
        self.pushButton_classifyIndex = self.logic.get('pushButton_classifyIndex')
        self.pushButton_preprocessNewData = self.logic.get('pushButton_preprocessNewData')
        self.pushButton_exportToClassify = self.logic.get('pushButton_exportToClassify')

        self.pushButton_trainNetwork = self.logic.get('pushButton_trainNetwork')
        self.pushButton_exportUntrainedNetwork = self.logic.get('pushButton_ExportUntrainedNetwork')
        self.pushButton_exportNetwork = self.logic.get('pushButton_ExportNetwork')
        self.pathLineEdit_CSVFileDataset = self.logic.get('pathLineEdit_CSVFileDataset')
        self.pathLineEdit_CSVFileMeansShape = self.logic.get('pathLineEdit_CSVFileMeansShape')
        self.pushButton_preprocessData = self.logic.get('pushButton_preprocessData')
        self.label_stateNetwork = self.logic.get('label_stateNetwork')

        self.pathLineEdit_networkPath = self.logic.get('ctkPathLineEdit_networkPath')

        self.collapsibleGroupBox_advancedParameters = self.logic.get('collapsibleGroupBox_advancedParameters')
        self.checkBox_features = self.logic.get('checkBox_features')
        self.checkableComboBox_choiceOfFeatures = self.logic.get('checkableComboBox_choiceOfFeatures')
        self.checkBox_numsteps = self.logic.get('checkBox_numsteps')
        self.spinBox_numsteps = self.logic.get('spinBox_numsteps')
        self.comboBox_controlGroup_features = self.logic.get('comboBox_controlGroup_features')
        self.checkBox_numberOfLayers = self.logic.get('checkBox_numberOfLayers')
        self.spinBox_numberOfLayers = self.logic.get('spinBox_numberOfLayers')

        #          Tab: Result / Analysis
        self.collapsibleButton_Result = self.logic.get('CollapsibleButton_Result')
        self.tableWidget_result = self.logic.get('tableWidget_result')
        self.pushButton_exportResult = self.logic.get('pushButton_exportResult')
        
        # Tab: Compute Average Groups
        self.CollapsibleButton_computeAverageGroups = self.logic.get('CollapsibleButton_computeAverageGroups')
        self.pathLineEdit_selectionClassificationGroups = self.logic.get('PathLineEdit_selectionClassificationGroups')
        self.pushButton_previewGroups = self.logic.get('pushButton_previewGroups')
        self.MRMLTreeView_classificationGroups = self.logic.get('MRMLTreeView_classificationGroups')
        self.directoryButton_exportMeanGroups = self.logic.get('directoryButton_exportMeanGroups')
        self.pushButton_exportMeanGroups = self.logic.get('pushButton_exportMeanGroups')
        self.pushButton_computeMeanGroup = self.logic.get('pushButton_computeMeanGroup')
        self.pathLineEdit_meanGroup = self.logic.get('pathLineEdit_meanGroup')

        # Widget Configuration

        #     disable/enable and hide/show widget
        self.comboBox_healthyGroup.setDisabled(True)
        self.pushButton_exportUpdatedClassification.setDisabled(True)
        self.checkableComboBox_ChoiceOfGroup.setDisabled(True)
        self.tableWidget_VTKFiles.setDisabled(True)
        self.pushButton_previewVTKFiles.setDisabled(True)

        self.pushButton_previewGroups.setDisabled(True)
        self.pushButton_computeMeanGroup.setDisabled(True)
        self.directoryButton_exportMeanGroups.setDisabled(True)
        self.pushButton_exportMeanGroups.setDisabled(True)

        self.pushButton_trainNetwork.setDisabled(True)
        self.pushButton_exportNetwork.setDisabled(True)
        self.pushButton_exportUntrainedNetwork.setDisabled(True)
        self.pushButton_preprocessData.setDisabled(True)

        self.pushButton_exportToClassify.setDisabled(True)
        self.pushButton_classifyIndex.setDisabled(True)

        self.label_stateNetwork.hide()

        self.collapsibleButton_classificationNetwork.setChecked(True)
        self.collapsibleButton_creationCSVFile.setChecked(False)
        self.collapsibleButton_previewClassificationGroups.setChecked(False)
        self.CollapsibleButton_computeAverageGroups.setChecked(False)
        self.collapsibleButton_Result.setChecked(False)
        self.collapsibleGroupBox_advancedParameters.setChecked(False)
        for ft in self.allFeatures:
            self.checkableComboBox_choiceOfFeatures.addItem(ft)


        #     qMRMLNodeComboBox configuration
        self.MRMLNodeComboBox_VTKInputData.setMRMLScene(slicer.mrmlScene)

        #     initialisation of the stackedWidget to display the button "add group"
        self.stackedWidget_manageGroup.setCurrentIndex(0)

        #     spinbox configuration in the tab "Creation of CSV File for Classification Groups"
        self.spinBox_group.setMinimum(0)
        self.spinBox_group.setMaximum(0)
        self.spinBox_group.setValue(0)

        #     spinbox configuration in the Advanced parameters
        self.spinBox_numsteps.setMinimum(11)
        self.spinBox_numsteps.setMaximum(10001)
        self.spinBox_numsteps.setValue(1001)

        self.spinBox_numberOfLayers.setMinimum(1)
        self.spinBox_numberOfLayers.setMaximum(2)
        self.spinBox_numberOfLayers.setValue(2)

        #     tree view configuration
        headerTreeView = self.MRMLTreeView_classificationGroups.header()
        headerTreeView.setVisible(False)
        self.MRMLTreeView_classificationGroups.setMRMLScene(slicer.app.mrmlScene())
        self.MRMLTreeView_classificationGroups.sortFilterProxyModel().nodeTypes = ['vtkMRMLModelNode']
        self.MRMLTreeView_classificationGroups.setDisabled(True)
        sceneModel = self.MRMLTreeView_classificationGroups.sceneModel()
        # sceneModel.setHorizontalHeaderLabels(["Group Classification"])
        sceneModel.colorColumn = 1
        sceneModel.opacityColumn = 2
        headerTreeView.setStretchLastSection(False)
        headerTreeView.setResizeMode(sceneModel.nameColumn,qt.QHeaderView.Stretch)
        headerTreeView.setResizeMode(sceneModel.colorColumn,qt.QHeaderView.ResizeToContents)
        headerTreeView.setResizeMode(sceneModel.opacityColumn,qt.QHeaderView.ResizeToContents)

        #     configuration of the table for preview VTK file
        self.tableWidget_VTKFiles.setColumnCount(4)
        self.tableWidget_VTKFiles.setHorizontalHeaderLabels([' VTK files ', ' Group ', ' Visualization ', 'Color'])
        self.tableWidget_VTKFiles.setColumnWidth(0, 200)
        horizontalHeader = self.tableWidget_VTKFiles.horizontalHeader()
        horizontalHeader.setStretchLastSection(False)
        horizontalHeader.setResizeMode(0,qt.QHeaderView.Stretch)
        horizontalHeader.setResizeMode(1,qt.QHeaderView.ResizeToContents)
        horizontalHeader.setResizeMode(2,qt.QHeaderView.ResizeToContents)
        horizontalHeader.setResizeMode(3,qt.QHeaderView.ResizeToContents)
        self.tableWidget_VTKFiles.verticalHeader().setVisible(True)

        #     configuration of the table to display the result
        self.tableWidget_result.setColumnCount(2)
        self.tableWidget_result.setHorizontalHeaderLabels([' VTK files ', ' Assigned Group '])
        self.tableWidget_result.setColumnWidth(0, 300)
        horizontalHeader = self.tableWidget_result.horizontalHeader()
        horizontalHeader.setStretchLastSection(False)
        horizontalHeader.setResizeMode(0,qt.QHeaderView.Stretch)
        horizontalHeader.setResizeMode(1,qt.QHeaderView.ResizeToContents)
        self.tableWidget_result.verticalHeader().setVisible(False)

        # --------------------------------------------------------- #
        #                       Connection                          #
        # --------------------------------------------------------- #
        #          Tab: Creation of CSV File for Classification Groups
        self.collapsibleButton_creationCSVFile.connect('clicked()',
                                                       lambda: self.onSelectedCollapsibleButtonOpen(self.collapsibleButton_creationCSVFile))
        self.spinBox_group.connect('valueChanged(int)', self.onManageGroup)
        self.pushButton_addGroup.connect('clicked()', self.onAddGroupForCreationCSVFile)
        self.pushButton_removeGroup.connect('clicked()', self.onRemoveGroupForCreationCSVFile)
        self.pushButton_modifyGroup.connect('clicked()', self.onModifyGroupForCreationCSVFile)
        self.pushButton_exportCSVfile.connect('clicked()', self.onExportForCreationCSVFile)
        # #          Tab: Preview / Update Classification Groups
        self.collapsibleButton_previewClassificationGroups.connect('clicked()',
                                                                    lambda: self.onSelectedCollapsibleButtonOpen(self.collapsibleButton_previewClassificationGroups))
        self.pathLineEdit_previewGroups.connect('currentPathChanged(const QString)', self.onSelectPreviewGroups)
        self.checkableComboBox_ChoiceOfGroup.connect('checkedIndexesChanged()', self.onCheckableComboBoxValueChanged)
        self.pushButton_previewVTKFiles.connect('clicked()', self.onPreviewVTKFiles)
        self.pushButton_exportUpdatedClassification.connect('clicked()', self.onExportUpdatedClassificationGroups)
       
        #          Tab: Select Input Data
        self.collapsibleButton_classificationNetwork.connect('clicked()',
                                                       lambda: self.onSelectedCollapsibleButtonOpen(self.collapsibleButton_classificationNetwork))
        self.MRMLNodeComboBox_VTKInputData.connect('currentNodeChanged(vtkMRMLNode*)', self.onVTKInputData)
        self.pathLineEdit_CSVInputData.connect('currentPathChanged(const QString)', self.onCSVInputData)
        self.pushButton_classifyIndex.connect('clicked()', self.onClassifyIndex)
        self.pushButton_preprocessNewData.connect('clicked()', self.onPreprocessNewData)
        self.pushButton_exportToClassify.connect('clicked()', self.onExportToClassify)

        self.checkableComboBox_choiceOfFeatures.connect('checkedIndexesChanged()', self.onCheckableComboBoxFeaturesChanged)

        #          Tab: Result / Analysis
        self.collapsibleButton_Result.connect('clicked()',
                                              lambda: self.onSelectedCollapsibleButtonOpen(self.collapsibleButton_Result))
        self.pushButton_exportResult.connect('clicked()', self.onExportResult)

        slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndCloseEvent, self.onCloseScene)

                 # Tab: Compute Average Groups
        self.CollapsibleButton_computeAverageGroups.connect('clicked()',
                                              lambda: self.onSelectedCollapsibleButtonOpen(self.CollapsibleButton_computeAverageGroups))

                 
        self.pathLineEdit_selectionClassificationGroups.connect('currentPathChanged(const QString)', self.onComputeAverageClassificationGroups)
        self.pushButton_previewGroups.connect('clicked()', self.onPreviewGroupMeans)
        self.pushButton_computeMeanGroup.connect('clicked()', self.onComputeMeanGroup)
        self.pushButton_exportMeanGroups.connect('clicked()', self.onExportMeanGroups)
        self.pathLineEdit_meanGroup.connect('currentPathChanged(const QString)', self.onMeanGroupCSV)

                # Tab: Classification Network
        self.pushButton_trainNetwork.connect('clicked()', self.onTrainNetwork)
        self.pushButton_exportNetwork.connect('clicked()', self.onExportNetwork)
        self.pushButton_exportUntrainedNetwork.connect('clicked()', self.onExportUntrainedNetwork)
        self.pathLineEdit_CSVFileDataset.connect('currentPathChanged(const QString)', self.onCSVFileDataset)
        self.pathLineEdit_CSVFileMeansShape.connect('currentPathChanged(const QString)', self.onCSVFileMeansShape)
        self.pushButton_preprocessData.connect('clicked()', self.onPreprocessData)
        self.stateCSVMeansShape = False
        self.stateCSVDataset = False
        self.pathLineEdit_networkPath.connect('currentPathChanged(const QString)', self.onNetworkPath)

    # function called each time that the user "enter" in Diagnostic Index interface
    def enter(self):
        #TODO
        pass

    # function called each time that the user "exit" in Diagnostic Index interface
    def exit(self):
        #TODO
        pass

    # function called each time that the scene is closed (if Diagnostic Index has been initialized)
    def onCloseScene(self, obj, event):

        numItem = self.comboBox_healthyGroup.count
        for i in range(0, numItem):
            self.comboBox_healthyGroup.removeItem(0)
            
        self.comboBox_healthyGroup.clear()

        print("onCloseScene")
        self.dictVTKFiles = dict()
        self.dictGroups = dict()
        self.dictCSVFile = dict()
        self.directoryList = list()
        self.groupSelected = set()
        self.dictShapeModels = dict()
        self.patientList = list()
        self.dictResults = dict()
        self.dictFeatData = dict()

        self.featuresList = list()
        
        # Tab: New Classification Groups
        self.pathLineEdit_previewGroups.setCurrentPath(" ")
        self.checkableComboBox_ChoiceOfGroup.setDisabled(True)
        self.tableWidget_VTKFiles.clear()
        self.tableWidget_VTKFiles.setColumnCount(4)
        self.tableWidget_VTKFiles.setHorizontalHeaderLabels([' VTK files ', ' Group ', ' Visualization ', 'Color'])
        self.tableWidget_VTKFiles.setColumnWidth(0, 200)
        horizontalHeader = self.tableWidget_VTKFiles.horizontalHeader()
        horizontalHeader.setStretchLastSection(False)
        horizontalHeader.setResizeMode(0,qt.QHeaderView.Stretch)
        horizontalHeader.setResizeMode(1,qt.QHeaderView.ResizeToContents)
        horizontalHeader.setResizeMode(2,qt.QHeaderView.ResizeToContents)
        horizontalHeader.setResizeMode(3,qt.QHeaderView.ResizeToContents)
        self.tableWidget_VTKFiles.verticalHeader().setVisible(False)
        self.tableWidget_VTKFiles.setDisabled(True)
        self.pushButton_previewVTKFiles.setDisabled(True)
        # le bt compute
        self.pushButton_exportUpdatedClassification.setDisabled(True)

        # Tab: Selection of Classification Groups
        self.pathLineEdit_selectionClassificationGroups.setCurrentPath(" ")
        if self.comboBox_healthyGroup.enabled:
            self.comboBox_healthyGroup.clear()
        self.comboBox_healthyGroup.setDisabled(True)

        # Tab: Preview of Classification Group
        self.MRMLTreeView_classificationGroups.setDisabled(True)
        self.pushButton_previewGroups.setDisabled(True)

        # Tab: Select Input Data
        self.pathLineEdit_CSVInputData.setCurrentPath(" ")
        self.pathLineEdit_CSVFileDataset.setCurrentPath(" ")
        self.pathLineEdit_CSVFileMeansShape.setCurrentPath(" ")
        self.pathLineEdit_networkPath.setCurrentPath(" ")
        self.pathLineEdit_meanGroup.setCurrentPath(" ")

        #          Tab: Result / Analysis
        self.collapsibleButton_Result = self.logic.get('CollapsibleButton_Result')
        self.tableWidget_result = self.logic.get('tableWidget_result')
        self.pushButton_exportResult = self.logic.get('pushButton_exportResult')
        
                 # Tab: Compute Average Groups
        self.CollapsibleButton_computeAverageGroups = self.logic.get('CollapsibleButton_computeAverageGroups')
        self.pathLineEdit_selectionClassificationGroups = self.logic.get('PathLineEdit_selectionClassificationGroups')
        self.pushButton_previewGroups = self.logic.get('pushButton_previewGroups')
        self.MRMLTreeView_classificationGroups = self.logic.get('MRMLTreeView_classificationGroups')
        self.directoryButton_exportMeanGroups = self.logic.get('directoryButton_exportMeanGroups')
        self.pushButton_exportMeanGroups = self.logic.get('pushButton_exportMeanGroups')
        self.pushButton_computeMeanGroup = self.logic.get('pushButton_computeMeanGroup')
        self.pathLineEdit_meanGroup = self.logic.get('pathLineEdit_meanGroup')

        # Tab: Result / Analysis
        self.tableWidget_result.clear()
        self.tableWidget_result.setColumnCount(2)
        self.tableWidget_result.setHorizontalHeaderLabels([' VTK files ', ' Assigned Group '])
        self.tableWidget_result.setColumnWidth(0, 300)
        horizontalHeader = self.tableWidget_result.horizontalHeader()
        horizontalHeader.setStretchLastSection(False)
        horizontalHeader.setResizeMode(0,qt.QHeaderView.Stretch)
        horizontalHeader.setResizeMode(1,qt.QHeaderView.ResizeToContents)
        self.tableWidget_result.verticalHeader().setVisible(False)

        # Enable/disable
        self.comboBox_healthyGroup.setDisabled(True)
        self.pushButton_exportUpdatedClassification.setDisabled(True)
        self.checkableComboBox_ChoiceOfGroup.setDisabled(True)
        self.tableWidget_VTKFiles.setDisabled(True)
        self.pushButton_previewVTKFiles.setDisabled(True)

        self.pushButton_previewGroups.setDisabled(True)
        self.pushButton_computeMeanGroup.setDisabled(True)
        self.directoryButton_exportMeanGroups.setDisabled(True)
        self.pushButton_exportMeanGroups.setDisabled(True)

        self.pushButton_trainNetwork.setDisabled(True)
        self.pushButton_exportNetwork.setDisabled(True)
        self.pushButton_exportUntrainedNetwork.setDisabled(True)
        self.pushButton_preprocessData.setDisabled(True)
        self.pushButton_exportToClassify.setDisabled(True)
        self.pushButton_classifyIndex.setDisabled(True)

        self.label_stateNetwork.hide()
        self.stateCSVMeansShape = False
        self.stateCSVDataset = False

        self.collapsibleButton_classificationNetwork.setChecked(True)
        self.collapsibleButton_creationCSVFile.setChecked(False)
        self.collapsibleButton_previewClassificationGroups.setChecked(False)
        self.CollapsibleButton_computeAverageGroups.setChecked(False)
        self.collapsibleButton_Result.setChecked(False)
        self.collapsibleGroupBox_advancedParameters.setChecked(False)
        self.comboBox_healthyGroup.clear()
        self.comboBox_controlGroup_features.clear()

        #     qMRMLNodeComboBox configuration
        self.MRMLNodeComboBox_VTKInputData.setMRMLScene(slicer.mrmlScene)

        #     initialisation of the stackedWidget to display the button "add group"
        self.stackedWidget_manageGroup.setCurrentIndex(0)

        #     spinbox configuration in the tab "Creation of CSV File for Classification Groups"
        self.spinBox_group.setMinimum(0)
        self.spinBox_group.setMaximum(0)
        self.spinBox_group.setValue(0)

        self.MRMLTreeView_classificationGroups.setMRMLScene(slicer.app.mrmlScene())


    def onSelectedCollapsibleButtonOpen(self, selectedCollapsibleButton):
        """  Only one tab can be display at the same time:
        When one tab is opened all the other tabs are closed 
        """
        if selectedCollapsibleButton.isChecked():
            collapsibleButtonList = [self.collapsibleButton_creationCSVFile,
                                     self.collapsibleButton_previewClassificationGroups,
                                     self.CollapsibleButton_computeAverageGroups,
                                     self.collapsibleButton_classificationNetwork,
                                     self.collapsibleButton_Result]
            for collapsibleButton in collapsibleButtonList:
                collapsibleButton.setChecked(False)
            selectedCollapsibleButton.setChecked(True)

    # ---------------------------------------------------- #
    # Tab: Creation of CSV File for Classification Groups  #
    # ---------------------------------------------------- #

    def onManageGroup(self):
        """ Function to display the 3 button:
            - "Add Group" for a group which hasn't been added yet
            - "Remove Group" for the last group added
            - "Modify Group" for all the groups added
        """
        if self.spinBox_group.maximum == self.spinBox_group.value:
            self.stackedWidget_manageGroup.setCurrentIndex(0)
        else:
            self.stackedWidget_manageGroup.setCurrentIndex(1)
            if (self.spinBox_group.maximum - 1) == self.spinBox_group.value:
                self.pushButton_removeGroup.show()
            else:
                self.pushButton_removeGroup.hide()
            # Update the path of the directory button
            if len(self.directoryList) > 0:
                self.directoryButton_creationCSVFile.directory = self.directoryList[self.spinBox_group.value - 1]

    def onAddGroupForCreationCSVFile(self):
        """Function to add a group of the dictionary
        - Add the paths of all the vtk files found in the directory given 
        of a dictionary which will be used to create the CSV file
        """
        # Error message
        directory = self.directoryButton_creationCSVFile.directory.encode('utf-8')
        if directory in self.directoryList:
            index = self.directoryList.index(directory) + 1
            slicer.util.errorDisplay('Path of directory already used for the group ' + str(index))
            return

        # Add the paths of vtk files of the dictionary
        self.logic.addGroupToDictionary(self.dictCSVFile, directory, self.directoryList, self.spinBox_group.value)
        condition = self.logic.checkSeveralMeshInDict(self.dictCSVFile)

        if not condition:
            # Remove the paths of vtk files of the dictionary
            self.logic.removeGroupToDictionary(self.dictCSVFile, self.directoryList, self.spinBox_group.value)
            return

        # Increment of the number of the group in the spinbox
        self.spinBox_group.blockSignals(True)
        self.spinBox_group.setMaximum(self.spinBox_group.value + 1)
        self.spinBox_group.setValue(self.spinBox_group.value + 1)
        self.spinBox_group.blockSignals(False)

        # Message for the user
        slicer.util.delayDisplay("Group Added")
        
    def onRemoveGroupForCreationCSVFile(self):
        """ Function to remove a group of the dictionary
            - Remove the paths of all the vtk files corresponding to the selected group 
            of the dictionary which will be used to create the CSV file
        """
        # Remove the paths of the vtk files of the dictionary
        self.logic.removeGroupToDictionary(self.dictCSVFile, self.directoryList, self.spinBox_group.value)

        # Decrement of the number of the group in the spinbox
        self.spinBox_group.blockSignals(True)
        self.spinBox_group.setMaximum(self.spinBox_group.maximum - 1)
        self.spinBox_group.blockSignals(False)

        # Change the buttons "remove group" and "modify group" in "add group"
        self.stackedWidget_manageGroup.setCurrentIndex(0)

        # Message for the user
        slicer.util.delayDisplay("Group removed")

    def onModifyGroupForCreationCSVFile(self):
        """ Function to modify a group of the dictionary:
            - Remove of the dictionary the paths of all vtk files corresponding to the selected group
            - Add of the dictionary the new paths of all the vtk files
        """
        # Error message
        directory = self.directoryButton_creationCSVFile.directory.encode('utf-8')
        if directory in self.directoryList:
            index = self.directoryList.index(directory) + 1
            slicer.util.errorDisplay('Path of directory already used for the group ' + str(index))
            return

        # Remove the paths of vtk files of the dictionary
        self.logic.removeGroupToDictionary(self.dictCSVFile, self.directoryList, self.spinBox_group.value)

        # Add the paths of vtk files of the dictionary
        self.logic.addGroupToDictionary(self.dictCSVFile, directory, self.directoryList, self.spinBox_group.value)

        # Message for the user
        slicer.util.delayDisplay("Group modified")

    def onExportForCreationCSVFile(self):
        """ Function to export the CSV file in the directory chosen by the user
            - Save the CSV file from the dictionary previously filled
            - Load automatically this CSV file in the next tab: "Creation of New Classification Groups"
        """
        # Path of the csv file
        dlg = ctk.ctkFileDialog()
        filepath = dlg.getSaveFileName(None, "Export CSV file for Classification groups", os.path.join(qt.QDir.homePath(), "Desktop"), "CSV File (*.csv)")

        directory = os.path.dirname(filepath)
        basename = os.path.basename(filepath)

        # Save the CSV File
        self.logic.creationCSVFile(directory, basename, self.dictCSVFile, "Groups")

        # Re-Initialization of the first tab
        self.spinBox_group.setMaximum(1)
        self.spinBox_group.setValue(1)
        self.stackedWidget_manageGroup.setCurrentIndex(0)
        self.directoryButton_creationCSVFile.directory = qt.QDir.homePath() + '/Desktop'

        # Re-Initialization of:
        #     - the dictionary containing all the paths of the vtk groups
        #     - the list containing all the paths of the different directories
        self.directoryList = list()
        self.dictCSVFile = dict()

        # Message in the python console
        print("Export CSV File: " + filepath)

        # Load automatically the CSV file in the pathline in the next tab "Creation of New Classification Groups"
        self.pathLineEdit_previewGroups.setCurrentPath(filepath)
        self.pathLineEdit_selectionClassificationGroups.setCurrentPath(filepath)
        self.pathLineEdit_CSVFileDataset.setCurrentPath(filepath)

    # ---------------------------------------------------- #
    #     Tab: Creation of New Classification Groups       #
    #     
    #     Preview/Update classification Groups
    #     
    # ---------------------------------------------------- #

    def onSelectPreviewGroups(self):
        """ Function to read the CSV file containing all the vtk 
        filepaths needed to create the new Classification Groups 
        """
        # Re-initialization of the dictionary containing all the vtk files
        # which will be used to create a new Classification Groups
        self.dictVTKFiles = dict()

        # Check if the path exists:
        if not os.path.exists(self.pathLineEdit_previewGroups.currentPath):
            return

        # print("------ Creation of a new Classification Groups ------")
        # Check if it's a CSV file
        condition1 = self.logic.checkExtension(self.pathLineEdit_previewGroups.currentPath, ".csv")
        if not condition1:
            self.pathLineEdit_previewGroups.setCurrentPath(" ")
            return

        # Download the CSV file
        self.logic.table = self.logic.readCSVFile(self.pathLineEdit_previewGroups.currentPath)
        condition2 = self.logic.creationDictVTKFiles(self.dictVTKFiles)
        condition3 = self.logic.checkSeveralMeshInDict(self.dictVTKFiles)

        # If the file is not conformed:
        #    Re-initialization of the dictionary containing all the data
        #    which will be used to create a new Classification Groups
        if not (condition2 and condition3):
            self.dictVTKFiles = dict()
            self.pathLineEdit_previewGroups.setCurrentPath(" ")
            return

        # Fill the table for the preview of the vtk files in Shape Population Viewer
        self.logic.fillTableForPreviewVTKFilesInSPV(self.dictVTKFiles,
                                               self.checkableComboBox_ChoiceOfGroup,
                                               self.tableWidget_VTKFiles)

        # Enable/disable buttons
        self.checkableComboBox_ChoiceOfGroup.setEnabled(True)
        self.tableWidget_VTKFiles.setEnabled(True)
        self.pushButton_previewVTKFiles.setEnabled(True)
        # self.pushButton_compute.setEnabled(True)

    
    def onCheckableComboBoxValueChanged(self):
        """ Function to manage the checkable combobox to allow 
        the user to choose the group that he wants to preview in SPV
        """
        # Update the checkboxes in the qtableWidget of each vtk file
        index = self.checkableComboBox_ChoiceOfGroup.currentIndex
        for row in range(0,self.tableWidget_VTKFiles.rowCount):
            # Recovery of the group of the vtk file contained in the combobox (column 2)
            widget = self.tableWidget_VTKFiles.cellWidget(row, 1)
            tuple = widget.children()
            comboBox = qt.QComboBox()
            comboBox = tuple[1]
            group = comboBox.currentIndex + 1
            if group == (index + 1):
                # check the checkBox
                widget = self.tableWidget_VTKFiles.cellWidget(row, 2)
                tuple = widget.children()
                checkBox = tuple[1]
                checkBox.blockSignals(True)
                item = self.checkableComboBox_ChoiceOfGroup.model().item(index, 0)
                if item.checkState():
                    checkBox.setChecked(True)
                    self.groupSelected.add(index + 1)
                else:
                    checkBox.setChecked(False)
                    self.groupSelected.discard(index + 1)
                checkBox.blockSignals(False)

        # Update the color in the qtableWidget of each vtk file
        colorTransferFunction = self.logic.creationColorTransfer(self.groupSelected)
        self.updateColorInTableForPreviewInSPV(colorTransferFunction)

    
    def onGroupValueChanged(self):
        """ Function to manage the combobox which 
        allow the user to change the group of a vtk file 
        """
        # Updade the dictionary which containing the VTK files sorted by groups
        self.logic.onComboBoxTableValueChanged(self.dictVTKFiles, self.tableWidget_VTKFiles)

        # Update the checkable combobox which display the groups selected to preview them in SPV
        self.onCheckBoxTableValueChanged()

        # Enable exportation of the last updated csv file
        self.pushButton_exportUpdatedClassification.setEnabled(True)
        # Default path to override the previous one
        # self.directoryButton_exportUpdatedClassification.directory = os.path.dirname(self.pathLineEdit_previewGroups.currentPath)


    def onCheckBoxTableValueChanged(self):
        """ Function to manage the checkbox in 
        the table used to make a preview in SPV 
        """
        self.groupSelected = set()
        # Update the checkable comboBox which allow to select what groups the user wants to display in SPV
        self.checkableComboBox_ChoiceOfGroup.blockSignals(True)
        allcheck = True
        for key, value in self.dictVTKFiles.items():
            item = self.checkableComboBox_ChoiceOfGroup.model().item(key, 0)
            if not value == []:
                for vtkFile in value:
                    filename = os.path.basename(vtkFile)
                    for row in range(0,self.tableWidget_VTKFiles.rowCount):
                        qlabel = self.tableWidget_VTKFiles.cellWidget(row, 0)
                        if qlabel.text == filename:
                            widget = self.tableWidget_VTKFiles.cellWidget(row, 2)
                            tuple = widget.children()
                            checkBox = tuple[1]
                            if not checkBox.checkState():
                                allcheck = False
                                item.setCheckState(0)
                            else:
                                self.groupSelected.add(key)
                if allcheck:
                    item.setCheckState(2)
            else:
                item.setCheckState(0)
            allcheck = True
        self.checkableComboBox_ChoiceOfGroup.blockSignals(False)

        # Update the color in the qtableWidget which will display in SPV
        colorTransferFunction = self.logic.creationColorTransfer(self.groupSelected)
        self.updateColorInTableForPreviewInSPV(colorTransferFunction)

    
    def updateColorInTableForPreviewInSPV(self, colorTransferFunction):
        """ Function to update the colors that the selected 
        vtk files will have in Shape Population Viewer
        """
        for row in range(0,self.tableWidget_VTKFiles.rowCount):
            # Recovery of the group display in the table for each vtk file
            widget = self.tableWidget_VTKFiles.cellWidget(row, 1)
            tuple = widget.children()
            comboBox = qt.QComboBox()
            comboBox = tuple[1]
            group = comboBox.currentIndex + 1

            # Recovery of the checkbox for each vtk file
            widget = self.tableWidget_VTKFiles.cellWidget(row, 2)
            tuple = widget.children()
            checkBox = qt.QCheckBox()
            checkBox = tuple[1]

            # If the checkbox is check, the color is found thanks to the color transfer function
            # Else the color is put at white
            if checkBox.isChecked():
                rgb = colorTransferFunction.GetColor(group)
                widget = self.tableWidget_VTKFiles.cellWidget(row, 3)
                self.tableWidget_VTKFiles.item(row,3).setBackground(qt.QColor(rgb[0]*255,rgb[1]*255,rgb[2]*255))
            else:
                self.tableWidget_VTKFiles.item(row,3).setBackground(qt.QColor(255,255,255))

    def onPreviewVTKFiles(self):
        """ Function to display the selected vtk files in Shape Population Viewer
            - Add a color map "DisplayClassificationGroup"
            - Launch the CLI ShapePopulationViewer
        """
        # print("--- Preview VTK Files in ShapePopulationViewer ---")
        if os.path.exists(self.pathLineEdit_previewGroups.currentPath):
            # Creation of a color map to visualize each group with a different color in ShapePopulationViewer
            self.logic.addColorMap(self.tableWidget_VTKFiles, self.dictVTKFiles)

            # Creation of a CSV file to load the vtk files in ShapePopulationViewer
            filePathCSV = slicer.app.temporaryPath + '/' + 'VTKFilesPreview_OAIndex.csv'
            self.logic.creationCSVFileForSPV(filePathCSV, self.tableWidget_VTKFiles, self.dictVTKFiles)

            # Launch the CLI ShapePopulationViewer
            parameters = {}
            parameters["CSVFile"] = filePathCSV
            module = slicer.modules.shapepopulationviewer
            slicer.cli.run(module, None, parameters, wait_for_completion=True)

            # Remove the vtk files previously created in the temporary directory of Slicer
            for value in self.dictVTKFiles.values():
                self.logic.removeDataVTKFiles(value)

    def onExportUpdatedClassificationGroups(self):
        """ Function to export the new Classification Groups
            - Data saved:
                - Save the mean vtk files in the selected directory
                - Save the CSV file in the selected directory
            - Load automatically the CSV file in the next tab: "Selection of Classification Groups"
        """
        # print("--- Export the new Classification Groups ---")

        dlg = ctk.ctkFileDialog()
        filepath = dlg.getSaveFileName(None, "Export Updated CSV file", "", "CSV File (*.csv)")

        directory = os.path.dirname(filepath)
        basename = os.path.basename(filepath)

        # Save the CSV File and the shape model of each group
        self.logic.creationCSVFile(directory, basename, self.dictVTKFiles, "Groups")

        # Re-Initialization of the dictionary containing the path of the shape model of each group
        # self.dictVTKFiles = dict()

        # Message for the user
        slicer.util.delayDisplay("Files Saved")

        # Disable the option to export the new data
        self.pushButton_exportUpdatedClassification.setDisabled(True)

        # Load automatically the CSV file in the pathline in the next tab "Selection of Classification Groups"
        if self.pathLineEdit_selectionClassificationGroups.currentPath == filepath:
            self.pathLineEdit_selectionClassificationGroups.setCurrentPath(" ")
        self.pathLineEdit_selectionClassificationGroups.setCurrentPath(filepath)
        self.pathLineEdit_CSVFileDataset.setCurrentPath(filepath)

    # ---------------------------------------------------- #
    #        Tab: Selection of Classification Groups       #
    #        
    #        Compute Average groups!!
    #        
    # ---------------------------------------------------- #

    
    def onComputeAverageClassificationGroups(self):
        """ Function to select the Classification Groups
        """
        # Re-initialization of the dictionary containing the Classification Groups
        self.dictShapeModels = dict()

        # Check if the path exists:
        if not os.path.exists(self.pathLineEdit_selectionClassificationGroups.currentPath):
            return

        # print("------ Selection of a Classification Groups ------")
        # Check if it's a CSV file
        condition1 = self.logic.checkExtension(self.pathLineEdit_selectionClassificationGroups.currentPath, ".csv")
        if not condition1:
            self.pathLineEdit_selectionClassificationGroups.setCurrentPath(" ")
            return


        # Read CSV File:
        self.logic.table = self.logic.readCSVFile(self.pathLineEdit_selectionClassificationGroups.currentPath)
        condition3 = self.logic.creationDictVTKFiles(self.dictShapeModels)
        condition2 = self.logic.checkSeveralMeshInDict(self.dictShapeModels)

        #    If the file is not conformed:
        #    Re-initialization of the dictionary containing the Classification Groups
        if not (condition2 and condition3):
            self.dictShapeModels = dict()
            self.pathLineEdit_selectionClassificationGroups.setCurrentPath(" ")
            return

        condition4 = self.logic.checkNumberOfPoints(self.dictShapeModels)
        if not condition4: 
            self.pathLineEdit_CSVFileDataset.setCurrentPath(" ")
            return
        self.pushButton_computeMeanGroup.setEnabled(True)

    def onComputeMeanGroup(self):
        """ Function to compute the average shape
        for each present group
        """
        # print("compute mean group")
        for group, listvtk in self.dictShapeModels.items():
            # Compute the mean of each group thanks to the CLI "computeMean"
            self.logic.computeMean(group, listvtk)

            # Storage of the means for each group
            self.logic.storageMean(self.dictGroups, group)

        self.pushButton_exportMeanGroups.setEnabled(True)
        self.directoryButton_exportMeanGroups.setEnabled(True)
        self.pushButton_previewGroups.setEnabled(True)
        
        self.pushButton_previewGroups.setEnabled(True)
        self.MRMLTreeView_classificationGroups.setEnabled(True)

        return 


    def onMeanGroupCSV(self):
        
        # print("------ onMeanGroupCSV ------")
        self.dictGroups = dict()

        # Check if it's a CSV file
        condition1 = self.logic.checkExtension(self.pathLineEdit_meanGroup.currentPath, ".csv")
        if not condition1:
            self.pathLineEdit_meanGroup.setCurrentPath(" ")
            return

        # Download the CSV file
        self.logic.table = self.logic.readCSVFile(self.pathLineEdit_meanGroup.currentPath)
        condition2 = self.logic.creationDictVTKFiles(self.dictGroups)
        condition3 = self.logic.checkOneMeshPerGroupInDict(self.dictGroups)

        # If the file is not conformed:
        #    Re-initialization of the dictionary containing all the data
        #    which will be used to create a new Classification Groups
        if not (condition2 and condition3):
            self.dictGroups = dict()
            return

        self.pushButton_previewGroups.setEnabled(True)
        self.comboBox_healthyGroup.setEnabled(True)
        self.comboBox_healthyGroup.clear()

        for key, value in self.dictGroups.items():
            # Fill the Checkable Combobox
            self.comboBox_healthyGroup.addItem("Group " + str(key))


    def onPreviewGroupMeans(self):
        """ Function to preview the Classification Groups in Slicer
            - The opacity of all the vtk files is set to 0.8
            - The healthy group is white and the others are red
        """
        # print("------ Preview of the Group's Mean in Slicer ------")

        list = slicer.mrmlScene.GetNodesByClass("vtkMRMLModelNode")
        end = list.GetNumberOfItems()
        for i in range(0,end):
            model = list.GetItemAsObject(i)
            if model.GetName()[:len("meanGroup")] == "meanGroup":
                hardenModel = slicer.mrmlScene.GetNodesByName(model.GetName()).GetItemAsObject(0)
                slicer.mrmlScene.RemoveNode(hardenModel)

        self.MRMLTreeView_classificationGroups.setMRMLScene(slicer.app.mrmlScene())
        self.MRMLTreeView_classificationGroups.setEnabled(True)

        for key in self.dictGroups.keys():
            filename = self.dictGroups.get(key, None)
            loader = slicer.util.loadModel
            loader(filename)

    # Change the color and the opacity for each vtk file
        list = slicer.mrmlScene.GetNodesByClass("vtkMRMLModelNode")
        end = list.GetNumberOfItems()
        for i in range(3,end):
            model = list.GetItemAsObject(i)
            disp = model.GetDisplayNode()
            # print ("model in color : " + str(model.GetName()))
            for group in self.dictGroups.keys():
                filename = self.dictGroups.get(group, None)
                if os.path.splitext(os.path.basename(filename))[0] == model.GetName():
                    if self.comboBox_healthyGroup.currentText == "Group " + str(group):
                        disp.SetColor(1, 1, 1)
                        disp.VisibilityOn()
                    else:
                        disp.SetColor(1, 0, 0)
                        disp.VisibilityOff()
                    disp.SetOpacity(0.8)
                    break
                disp.VisibilityOff()

        # Center the 3D view of the scene
        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(0)
        threeDView = threeDWidget.threeDView()
        threeDView.resetFocalPoint()


    def onExportMeanGroups(self):
        """ Function to export the computed average shapes 
        (VTK files) + a CSV file listing them 
        """
        # print("--- Export all the mean shapes + csv file ---")

        # Message for the user if files already exist
        directory = self.directoryButton_exportMeanGroups.directory.encode('utf-8')
        messageBox = ctk.ctkMessageBox()
        messageBox.setWindowTitle(' /!\ WARNING /!\ ')
        messageBox.setIcon(messageBox.Warning)
        filePathExisting = list()

        #   Check if the CSV file exists
        CSVfilePath = directory + "/MeanGroups.csv"
        if os.path.exists(CSVfilePath):
            filePathExisting.append(CSVfilePath)

        #   Check if the shape model exist
        for key, value in self.dictGroups.items():
            modelFilename = os.path.basename(value)
            modelFilePath = directory + '/' + modelFilename
            if os.path.exists(modelFilePath):
                filePathExisting.append(modelFilePath)

        #   Write the message for the user
        if len(filePathExisting) > 0:
            if len(filePathExisting) == 1:
                text = 'File ' + filePathExisting[0] + ' already exists!'
                informativeText = 'Do you want to replace it ?'
            elif len(filePathExisting) > 1:
                text = 'These files are already exist: \n'
                for path in filePathExisting:
                    text = text + path + '\n'
                    informativeText = 'Do you want to replace them ?'
            messageBox.setText(text)
            messageBox.setInformativeText(informativeText)
            messageBox.setStandardButtons( messageBox.No | messageBox.Yes)
            choice = messageBox.exec_()
            if choice == messageBox.No:
                return

        # Save the CSV File and the mean shape of each group
        dictForCSV = dict()
        for key, value in self.dictGroups.items():
            # Save the shape model (h5 file) of each group
            vtkbasename = os.path.basename(value)
            oldvtkpath = slicer.app.temporaryPath + "/" + vtkbasename
            newvtkpath = directory + "/" + vtkbasename
            shutil.copyfile(oldvtkpath, newvtkpath)
            dictForCSV[key] = newvtkpath

        # Save the CSV file containing all the data useful in order to compute OAIndex of a patient
        self.logic.creationCSVFile(directory, "MeanGroups.csv", dictForCSV, "MeanGroup")

        # Remove the shape model (GX.h5) of each group
        self.logic.removeDataAfterNCG(self.dictGroups)

        # Re-Initialization of the dictionary containing the path of the shape model of each group
        self.dictGroups = dictForCSV

        # Message for the user
        slicer.util.delayDisplay("Files Saved")
        print("Saved in :: " + directory + "/MeanGroups.csv")

        # Load automatically the CSV file in the pathline in the next tab "Selection of Classification Groups"
        if self.pathLineEdit_meanGroup.currentPath == CSVfilePath:
            self.pathLineEdit_meanGroup.setCurrentPath(" ")
        self.pathLineEdit_meanGroup.setCurrentPath(CSVfilePath)
        self.pathLineEdit_CSVFileMeansShape.setCurrentPath(CSVfilePath)

        return 


    # ---------------------------------------------------- #
    #               Tab: Select Input Data
    #               
    #               Classification Network                 #
    # ---------------------------------------------------- #

    def enableNetwork(self):
        """ Function to enable to train the Network if both 
        the Mean shape CSV file and classification group CSV file
        have been accepted
        """
        if self.stateCSVDataset and self.stateCSVMeansShape:
            self.pushButton_preprocessData.setEnabled(True)
        else:
            self.pushButton_trainNetwork.setDisabled(True)
            self.pushButton_preprocessData.setDisabled(True)
        return

    def onCSVFileDataset(self):
        """ Function to load the shapes listed in 
        the classification group CSV file
        """
        # Re-initialization of the dictionary containing the Training dataset
        self.dictShapeModels = dict()

        # Check if the path exists:
        if not os.path.exists(self.pathLineEdit_CSVFileDataset.currentPath):
            self.stateCSVDataset = False
            self.enableNetwork()
            return

        # print("------ Selection of a Dataset ------")
        # Check if it's a CSV file
        condition1 = self.logic.checkExtension(self.pathLineEdit_CSVFileDataset.currentPath, ".csv")
        if not condition1:
            self.pathLineEdit_CSVFileDataset.setCurrentPath(" ")
            self.stateCSVDataset = False
            self.enableNetwork()
            return

        # Read CSV File:
        self.logic.table = self.logic.readCSVFile(self.pathLineEdit_CSVFileDataset.currentPath)
        condition3 = self.logic.creationDictVTKFiles(self.dictShapeModels)
        condition2 = self.logic.checkSeveralMeshInDict(self.dictShapeModels)

        #    If the file is not conformed:
        #    Re-initialization of the dictionary containing the Classification Groups
        if not (condition2 and condition3):
            self.dictShapeModels = dict()
            self.pathLineEdit_CSVFileDataset.setCurrentPath(" ")
            self.stateCSVDataset = False
            self.enableNetwork()
            return

        # Condition : All the shapes should have the same number of points
        condition4 = self.logic.checkNumberOfPoints(self.dictShapeModels)
        if not condition4: 
            self.pathLineEdit_CSVFileDataset.setCurrentPath(" ")
            return

        self.stateCSVDataset = True
        self.enableNetwork()

        return


    def onCSVFileMeansShape(self):
        """ Function to load the shapes listed in 
        the mean shapes CSV file
        """
        # Re-initialization of the dictionary containing the Training dataset
        self.dictGroups = dict()

        # Check if the path exists:
        if not os.path.exists(self.pathLineEdit_CSVFileMeansShape.currentPath):
            self.stateCSVMeansShape = False
            self.enableNetwork()
            return

        # print("------ Selection of a Dataset ------")
        # Check if it's a CSV file
        condition1 = self.logic.checkExtension(self.pathLineEdit_CSVFileMeansShape.currentPath, ".csv")
        if not condition1:
            self.pathLineEdit_CSVFileMeansShape.setCurrentPath(" ")
            self.stateCSVMeansShape = False
            self.enableNetwork()
            return

        # Read CSV File:
        self.logic.table = self.logic.readCSVFile(self.pathLineEdit_CSVFileMeansShape.currentPath)
        condition3 = self.logic.creationDictVTKFiles(self.dictGroups)
        condition2 = self.logic.checkOneMeshPerGroupInDict(self.dictGroups)

        #    If the file is not conformed:
        #    Re-initialization of the dictionary containing the Classification Groups
        if not (condition2 and condition3):
            self.dictGroups = dict()
            self.pathLineEdit_CSVFileMeansShape.setCurrentPath(" ")
            self.stateCSVMeansShape = False
            self.enableNetwork()
            return

        # First condition : All the shapes should have the same number of points
        condition4 = self.logic.checkNumberOfPoints(self.dictGroups)
        if not condition4: 
            self.pathLineEdit_CSVFileMeansShape.setCurrentPath(" ")
            return

        self.stateCSVMeansShape = True
        self.comboBox_controlGroup_features.clear()
        for key, value in self.dictGroups.items():
            self.comboBox_controlGroup_features.addItem(str(key))

        self.enableNetwork()
        return


    def onCheckableComboBoxFeaturesChanged(self):
        """ Function to manage the features choosen by the user
        to base the neural network on
        """
        # print("----- Features check combo box changed -----")
        index =  self.checkableComboBox_choiceOfFeatures.currentIndex

        item = self.checkableComboBox_choiceOfFeatures.model().item(index, 0)
        print(str(item.text()))
        if item.checkState():
            if not self.featuresList.count(item.text()):
                self.featuresList.append(str(item.text()))
        else:
            if self.featuresList.count(item.text()):
                self.featuresList.remove(item.text())

        return


    def onPreprocessData(self):
        """ Function to prepare all the data before training the network
            - Extract all the features (CLI extractfeatures)
            - load only the features selected (class input_Data)
            - pickle all the dataset for the network
            - Create a zipfile with every file needed for the network
        """
        # print("----- onPreprocessData -----")
        self.dictFeatData = dict()
        self.pickle_file = ""

        tempPath = slicer.app.temporaryPath
        
        outputDir = os.path.join(tempPath, "dataFeatures")
        if os.path.isdir(outputDir):
            shutil.rmtree(outputDir)
        os.mkdir(outputDir) 

        #
        # Extract features on shapes, with SurfaceFeaturesExtractor
        meansList = ""
        for k, v in self.dictGroups.items():
            if meansList == "":
                meansList = str(v)
            else:
                meansList = meansList + "," +  str(v)

        # print(self.dictShapeModels)
        for group, listvtk in self.dictShapeModels.items():
            for shape in listvtk:
                self.logic.extractFeatures(shape, meansList, outputDir, train = True)

                # # Storage of the means for each group
                self.logic.storageFeaturesData(self.dictFeatData, self.dictShapeModels)

        # 
        # Pickle the data for the network
        ft_list = self.allFeatures
        self.controlAverage = None
        if self.collapsibleGroupBox_advancedParameters.checked and self.checkBox_features.checked:
            ft_list = self.featuresList
            if not len(ft_list):
                slicer.util.delayDisplay("Missing features")
                return

        if ft_list.count('Distance to control group') and ft_list.count('Distances to average shapes'): 
        # if both checked, remove distance to control, otherwise the feature will be used twice
            ft_list.remove('Distance to control group')
        else:
            self.controlAverage = int(str(self.comboBox_controlGroup_features.currentText))

        self.pickle_file = self.logic.pickleData(self.dictFeatData, ft_list, self.controlAverage)
        
        #
        # Zipping JSON + PICKLE files in one ZIP file
        # 
        tempPath = slicer.app.temporaryPath
        networkDir = os.path.join(tempPath, 'Network')

        meanGroupsDir = os.path.join(networkDir, 'meanGroups')
        os.mkdir(meanGroupsDir) 
        dictMeanGroups = dict()
        for group, file in self.dictGroups.items():
            # Copy the meanShapes into the NetworkDir
            shutil.copyfile(file, os.path.join(meanGroupsDir, os.path.basename(file)))
            dictMeanGroups[group] = os.path.basename(file)

        with open(os.path.join(meanGroupsDir,'meanGroups.json'), 'w') as f:
            json.dump(dictMeanGroups, f, ensure_ascii=False, indent = 4)

        # Zipper tout ca 
        self.archiveName = shutil.make_archive(base_name = networkDir, format = 'zip', root_dir = tempPath, base_dir = 'Network')
        self.pushButton_trainNetwork.setEnabled(True)
        self.pushButton_exportUntrainedNetwork.setEnabled(True)

        return

    def onExportUntrainedNetwork(self):
        """ Function to export the neural netowrk as
        a zipfile for later reuse
        """
        # print("----- onExportUntrainedNetwork -----")

        num_steps = 1001
        num_layers = 2

        if self.collapsibleGroupBox_advancedParameters.checked:
            if self.checkBox_numsteps.checked:
                num_steps = self.spinBox_numsteps.value
            if self.checkBox_numberOfLayers.checked:
                num_layers = self.spinBox_numberOfLayers.value
        # Path of the csv file
        dlg = ctk.ctkFileDialog()
        filepath = dlg.getSaveFileName(None, "Export Classification neural network", os.path.join(qt.QDir.homePath(), "Desktop"), "Archive Zip (*.zip)")

        self.logic.exportUntrainedNetwork(self.archiveName, filepath, num_steps = num_steps, num_layers = num_layers)
        return


    def onTrainNetwork(self):
        """ Function to call the logic function related 
        to the training of the neural network
        """
        # print("----- onTrainNetwork -----")
        self.label_stateNetwork.text = 'Computation running...'
        self.label_stateNetwork.show()

        num_steps = 1001
        num_layers = 2

        if self.collapsibleGroupBox_advancedParameters.checked:
            if self.checkBox_numsteps.checked:
                num_steps = self.spinBox_numsteps.value
            if self.checkBox_numberOfLayers.checked:
                num_layers = self.spinBox_numberOfLayers.value
        
        accuracy = self.logic.trainNetworkClassification(self.archiveName, num_steps = num_steps, num_layers = num_layers)
        
        print("ESTIMATED ACCURACY :: " + str(accuracy))

        self.label_stateNetwork.text = ("Estimated accuracy: %.1f%%" % accuracy)
        self.pushButton_exportNetwork.setEnabled(True)

        return

    def onExportNetwork(self):
        """ Function to export the neural netowrk as
        a zipfile for later reuse
        """
        # print("----- onExportNetwork -----")

        # Path of the csv file
        dlg = ctk.ctkFileDialog()
        filepath = dlg.getSaveFileName(None, "Export Classification neural network", os.path.join(qt.QDir.homePath(), "Desktop"), "Archive Zip (*.zip)")

        directory = os.path.dirname(filepath)
        networkpath = filepath.split(".zip",1)[0]

        self.logic.exportModelNetwork(networkpath)
        self.pathLineEdit_networkPath.currentPath = filepath
        self.label_stateNetwork.hide()
        return

    def onNetworkPath(self):
        """ Function to launch the network loading 
        when specifying the path to the zipfile
        """
        # print("----- onNetworkPath -----")
        condition1 = self.logic.checkExtension(self.pathLineEdit_networkPath.currentPath, '.zip')
        if not condition1:
            self.pathLineEdit_networkPath.setCurrentPath(" ")
            return

        self.dictGroups = dict()
        validModel = self.logic.validModelNetwork(self.pathLineEdit_networkPath.currentPath, self.dictGroups)

        # print(self.dictGroups)
        if not validModel:
            print("Error: Classifier not valid")
            self.pathLineEdit_networkPath.currentPath = ""
        else: 
            print("Network accepted")

        return

    
    def onVTKInputData(self):
        """ Function to select the vtk Input Data
        """
        # Remove the old vtk file in the temporary directory of slicer if it exists
        if self.patientList:
            print("onVTKInputData remove old vtk file")
            oldVTKPath = os.path.join(slicer.app.temporaryPath,os.path.basename(self.patientList[0]))
            if os.path.exists(oldVTKPath):
                os.remove(oldVTKPath)
        # print(self.patientList)
        # Re-Initialization of the patient list
        self.patientList = list()

        # Delete the path in CSV file
        currentNode = self.MRMLNodeComboBox_VTKInputData.currentNode()
        if currentNode == None:
            return
        self.pathLineEdit_CSVInputData.setCurrentPath(" ")

        # Adding the vtk file to the list of patient
        currentNode = self.MRMLNodeComboBox_VTKInputData.currentNode()
        if not currentNode == None:
            #     Save the selected node in the temporary directory of slicer
            vtkfilepath = os.path.join(slicer.app.temporaryPath, self.MRMLNodeComboBox_VTKInputData.currentNode().GetName() + ".vtk")
            self.logic.saveVTKFile(self.MRMLNodeComboBox_VTKInputData.currentNode().GetPolyData(), vtkfilepath)
            #     Adding to the list
            self.patientList.append(vtkfilepath)
        print(self.patientList)

    
    def onCSVInputData(self):
        """ Function to select the CSV Input Data
        """
        self.patientList = list()
        # Delete the path in VTK file
        if not os.path.exists(self.pathLineEdit_CSVInputData.currentPath):
            return
        self.MRMLNodeComboBox_VTKInputData.setCurrentNode(None)

        # Adding the name of the node a list
        if os.path.exists(self.pathLineEdit_CSVInputData.currentPath):
            patientTable = vtk.vtkTable
            patientTable = self.logic.readCSVFile(self.pathLineEdit_CSVInputData.currentPath)
            for i in range(0, patientTable.GetNumberOfRows()):
                self.patientList.append(patientTable.GetValue(i,0).ToString())
        
    def onPreprocessNewData(self):
        """ Function to preprocess the data to classify,
        independently of the classification. 
        - Extract Features
        - Pickle data
        """
        # print("------ Preprocess New Data ------")
        if self.MRMLNodeComboBox_VTKInputData.currentNode() == None and not self.pathLineEdit_CSVInputData.currentPath:
            slicer.util.errorDisplay('Miss the Input Data')
            return

        # *** Define the group type of a patient ***
        self.dictFeatData = dict()
        tempPath = slicer.app.temporaryPath
        networkDir = os.path.join(tempPath, 'Network')

        outputDir = os.path.join(tempPath, "dataToClassify")
        if os.path.isdir(outputDir):
            shutil.rmtree(outputDir)
        os.mkdir(outputDir) 

        #
        # Extract features on shapes, with SurfaceFeaturesExtractor and get new path (in slicer temp path)
        meansList = ""
        for k, v in self.dictGroups.items():
            if meansList == "":
                meansList = str(v)
            else:
                meansList = meansList + "," +  str(v)

        for shape in self.patientList:
            # Extract features de la/les shapes a classifier
            self.logic.extractFeatures(shape, meansList, outputDir, train = False)

        # Change paths in patientList to have shape with features
        self.logic.storageDataToClassify(self.dictFeatData, self.patientList, outputDir)

        pickleToClassify = self.logic.pickleToClassify(self.patientList, os.path.join(slicer.app.temporaryPath,'Network'))
        self.archiveName = shutil.make_archive(base_name = networkDir, format = 'zip', root_dir = tempPath, base_dir = 'Network')

        self.pushButton_exportToClassify.setEnabled(True)
        self.pushButton_classifyIndex.setEnabled(True)
        return


    def onExportToClassify(self):
        """ Function to extract the classifier and the data to classify
        Possibility to run remotely with this zip file 
        """
        tempPath = slicer.app.temporaryPath
        networkDir = os.path.join(tempPath, 'Network')

        dlg = ctk.ctkFileDialog()
        filepath = dlg.getSaveFileName(None, "Export Network and shapes to classify", os.path.join(qt.QDir.homePath(), "Desktop"), "Archive Zip (*.zip)")

        directory = os.path.dirname(filepath)
        path = filepath.split(".zip",1)[0]

        shutil.make_archive(path, 'zip', networkDir)

        return
        
    def onClassifyIndex(self):
        """ Function classify shapes
            - preprocess (extract features) the data
            - generate a complete zipfile for the network
        """
        # print("------ Compute the OA index Type of a patient ------")
        tempPath = slicer.app.temporaryPath
        networkDir = os.path.join(tempPath, 'Network')

        self.dictResults = dict()
        self.dictResults = self.logic.evalClassification(networkDir + ".zip")
        self.displayResult(self.dictResults)
        return

    # ---------------------------------------------------- #
    #               Tab: Result / Analysis                 #
    # ---------------------------------------------------- #

    def displayResult(self, dictResults):
        """ Function to display the result in a table
        """
        for VTKfilename, resultGroup in dictResults.items():
            row = self.tableWidget_result.rowCount
            self.tableWidget_result.setRowCount(row + 1)
            # Column 0: VTK file
            labelVTKFile = qt.QLabel(os.path.basename(VTKfilename))
            labelVTKFile.setAlignment(0x84)
            self.tableWidget_result.setCellWidget(row, 0, labelVTKFile)
            # Column 1: Assigned Group
            labelAssignedGroup = qt.QLabel(resultGroup)
            labelAssignedGroup.setAlignment(0x84)
            self.tableWidget_result.setCellWidget(row, 1, labelAssignedGroup)

        # open the results tab
        self.collapsibleButton_Result.setChecked(True)
        self.onSelectedCollapsibleButtonOpen(self.collapsibleButton_Result)


    def onExportResult(self):
        """ Function to export the result in a CSV File
        """
        # Path of the csv file
        dlg = ctk.ctkFileDialog()
        filepath = dlg.getSaveFileName(None, "Export CSV file for Classification groups", os.path.join(qt.QDir.homePath(), "Desktop"), "CSV File (*.csv)")

        directory = os.path.dirname(filepath)
        basename = os.path.basename(filepath)

        # Store data in a dictionary
        self.logic.creationCSVFileForResult(self.tableWidget_result, directory, basename)

        # Message in the python console and for the user
        print("Export CSV File: " + filepath)
        slicer.util.delayDisplay("Result saved")


# ------------------------------------------------------------------------------------ #
#                                   ALGORITHM                                          #
# ------------------------------------------------------------------------------------ #

class ShapeVariationAnalyzerLogic(ScriptedLoadableModuleLogic):
    def __init__(self, interface):
        self.interface = interface
        self.table = vtk.vtkTable
        self.colorBar = {'Point1': [0, 0, 1, 0], 'Point2': [0.5, 1, 1, 0], 'Point3': [1, 1, 0, 0]}
        self.input_Data = inputData.inputData()

    
    def get(self, objectName):
        """ Functions to recovery the widget in the .ui file
        """
        return slicer.util.findChild(self.interface.widget, objectName)

    def addGroupToDictionary(self, dictCSVFile, directory, directoryList, group):
        """ Function to add all the vtk filepaths 
        found in the given directory of a dictionary
        """
        # Fill a dictionary which contains the vtk files for the classification groups sorted by group
        valueList = list()
        for file in os.listdir(directory):
            if file.endswith(".vtk"):
                filepath = directory + '/' + file
                valueList.append(filepath)
        dictCSVFile[group] = valueList

        # Add the path of the directory
        directoryList.insert((group - 1), directory)

    def removeGroupToDictionary(self, dictCSVFile, directoryList, group):
        """ Function to remove the group of the dictionary
        """
        # Remove the group from the dictionary
        dictCSVFile.pop(group, None)

        # Remove the path of the directory
        directoryList.pop(group - 1)

    
    def checkExtension(self, filename, extension):
        """ Check if the path given has the right extension
        """
        if os.path.splitext(os.path.basename(filename))[1] == extension : 
            return True
        elif os.path.basename(filename) == "" or os.path.basename(filename) == " " :
            return False
        slicer.util.errorDisplay('Wrong extension file, a ' + extension + ' file is needed!')
        return False

    def readCSVFile(self, filename):
        """ Function to read a CSV file
        """
        print("CSV FilePath: " + filename)
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
                slicer.util.errorDisplay('VTK file not found, path not good at lign ' + str(i+2))
                return False
            value = dict.get(self.table.GetValue(i,1).ToInt(), None)
            if value == None:
                dict[self.table.GetValue(i,1).ToInt()] = self.table.GetValue(i,0).ToString()
            else:
                if type(value) is ListType:
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
            if type(value) is not ListType or len(value) == 1:
                slicer.util.errorDisplay('The group ' + str(key) + ' must contain more than one mesh.')
                return False
        return True

    def checkOneMeshPerGroupInDict(self, dict):
        """ Function to check if in each group 
        there is at least more than one mesh 
        """
        for key, value in dict.items():
            if type(value) is ListType: # or len(value) != 1:
                slicer.util.errorDisplay('The group ' + str(key) + ' must contain exactly one mesh.')
                return False
        return True

    def creationDictShapeModel(self, dict):
        """Function to store the shape models for each group in a dictionary
        The function return True IF
            - all the paths exist
            - the extension of the paths is .h5
            - there are only one shape model per group
        else False
        """
        for i in range(0, self.table.GetNumberOfRows()):
            if not os.path.exists(self.table.GetValue(i,0).ToString()):
                slicer.util.errorDisplay('VTK file not found, path not good at lign ' + str(i+2))
                return False
            if not os.path.splitext(os.path.basename(self.table.GetValue(i,0).ToString()))[1] == '.vtk':
                slicer.util.errorDisplay('Wrong extension file at lign ' + str(i+2) + '. A vtk file is needed!')
                return False
            dict[self.table.GetValue(i,1).ToInt()] = self.table.GetValue(i,0).ToString()

        return True

    def addColorMap(self, table, dictVTKFiles):
        """ Function to add a color map "DisplayClassificationGroup" 
        to all the vtk files which allow the user to visualize each 
        group with a different color in ShapePopulationViewer
        """
        for key, value in dictVTKFiles.items():
            for vtkFile in value:
                # Read VTK File
                reader = vtk.vtkDataSetReader()
                reader.SetFileName(vtkFile)
                reader.ReadAllVectorsOn()
                reader.ReadAllScalarsOn()
                reader.Update()
                polyData = reader.GetOutput()

                # Copy of the polydata
                polyDataCopy = vtk.vtkPolyData()
                polyDataCopy.DeepCopy(polyData)
                pointData = polyDataCopy.GetPointData()

                # Add a New Array "DisplayClassificationGroup" to the polydata copy
                # which will have as the value for all the points the group associated of the mesh
                numPts = polyDataCopy.GetPoints().GetNumberOfPoints()
                arrayName = "DisplayClassificationGroup"
                hasArrayInt = pointData.HasArray(arrayName)
                if hasArrayInt == 1:
                    pointData.RemoveArray(arrayName)
                arrayToAdd = vtk.vtkDoubleArray()
                arrayToAdd.SetName(arrayName)
                arrayToAdd.SetNumberOfComponents(1)
                arrayToAdd.SetNumberOfTuples(numPts)
                for i in range(0, numPts):
                    arrayToAdd.InsertTuple1(i, key)
                pointData.AddArray(arrayToAdd)

                # Save in the temporary directory in Slicer the vtk file with the new array
                # to visualize them in Shape Population Viewer
                writer = vtk.vtkPolyDataWriter()
                filepath = slicer.app.temporaryPath + '/' + os.path.basename(vtkFile)
                writer.SetFileName(filepath)
                if vtk.VTK_MAJOR_VERSION <= 5:
                    writer.SetInput(polyDataCopy)
                else:
                    writer.SetInputData(polyDataCopy)
                writer.Update()
                writer.Write()

    def creationCSVFileForSPV(self, filename, table, dictVTKFiles):
        """ Function to create a CSV file containing all the 
        selected vtk files that the user wants to display in SPV 
        """
        # Creation a CSV file with a header 'VTK Files'
        file = open(filename, 'w')
        cw = csv.writer(file, delimiter=',')
        cw.writerow(['VTK Files'])

        # Add the path of the vtk files if the users selected it
        for row in range(0,table.rowCount):
            # check the checkBox
            widget = table.cellWidget(row, 2)
            tuple = widget.children()
            checkBox = qt.QCheckBox()
            checkBox = tuple[1]
            if checkBox.isChecked():
                # Recovery of group fo each vtk file
                widget = table.cellWidget(row, 1)
                tuple = widget.children()
                comboBox = qt.QComboBox()
                comboBox = tuple[1]
                group = comboBox.currentIndex + 1
                # Recovery of the vtk filename
                qlabel = table.cellWidget(row, 0)
                vtkFile = qlabel.text
                pathVTKFile = slicer.app.temporaryPath + '/' + vtkFile
                cw.writerow([pathVTKFile])
        file.close()

    def fillTableForPreviewVTKFilesInSPV(self, dictVTKFiles, checkableComboBox, table):
        """Function to fill the table of the preview of all VTK files
            - Checkable combobox: allow the user to select one or several groups that he wants to display in SPV
            - Column 0: filename of the vtk file
            - Column 1: combobox with the group corresponding to the vtk file
            - Column 2: checkbox to allow the user to choose which models will be displayed in SPV
            - Column 3: color that the mesh will have in SPV
        """
        row = 0
        for key, value in dictVTKFiles.items():
            # Fill the Checkable Combobox
            checkableComboBox.addItem("Group " + str(key))
            # Table:
            for vtkFile in value:
                table.setRowCount(row + 1)
                # Column 0:
                filename = os.path.basename(vtkFile)
                labelVTKFile = qt.QLabel(filename)
                labelVTKFile.setAlignment(0x84)
                table.setCellWidget(row, 0, labelVTKFile)

                # Column 1:
                widget = qt.QWidget()
                layout = qt.QHBoxLayout(widget)
                comboBox = qt.QComboBox()
                comboBox.addItems(dictVTKFiles.keys())        
                comboBox.setCurrentIndex(key)
                layout.addWidget(comboBox)
                layout.setAlignment(0x84)
                layout.setContentsMargins(0, 0, 0, 0)
                widget.setLayout(layout)
                table.setCellWidget(row, 1, widget)
                comboBox.connect('currentIndexChanged(int)', self.interface.onGroupValueChanged)

                # Column 2:
                widget = qt.QWidget()
                layout = qt.QHBoxLayout(widget)
                checkBox = qt.QCheckBox()
                layout.addWidget(checkBox)
                layout.setAlignment(0x84)
                layout.setContentsMargins(0, 0, 0, 0)
                widget.setLayout(layout)
                table.setCellWidget(row, 2, widget)
                checkBox.connect('stateChanged(int)', self.interface.onCheckBoxTableValueChanged)

                # Column 3:
                table.setItem(row, 3, qt.QTableWidgetItem())
                table.item(row,3).setBackground(qt.QColor(255,255,255))

                row = row + 1

    def onComboBoxTableValueChanged(self, dictVTKFiles, table):
        """ Function to change the group of a vtk file
            - The user can change the group thanks to the combobox in the table used for the preview in SPV
        """
        # For each row of the table
        for row in range(0,table.rowCount):
            # Recovery of the group associated to the vtk file which is in the combobox
            widget = table.cellWidget(row, 1)
            tuple = widget.children()
            comboBox = qt.QComboBox()
            comboBox = tuple[1]
            group = comboBox.currentIndex
            # Recovery of the filename of vtk file
            qlabel = table.cellWidget(row, 0)
            vtkFile = qlabel.text

            # Update the dictionary if the vtk file has not the same group in the combobox than in the dictionary
            value = dictVTKFiles.get(group, None)
            if not any(vtkFile in s for s in value):
                # Find which list of the dictionary the vtk file is in
                for value in dictVTKFiles.values():
                    if any(vtkFile in s for s in value):
                        pathList = [s for s in value if vtkFile in s]
                        path = pathList[0]
                        # Remove the vtk file from the wrong group
                        value.remove(path)
                        # Add the vtk file in the right group
                        newvalue = dictVTKFiles.get(group, None)
                        newvalue.append(path)
                        break

    def creationColorTransfer(self, groupSelected):
        """ Function to create the same color transfer function than there is in SPV
        """
        # Creation of the color transfer function with the updated range
        colorTransferFunction = vtk.vtkColorTransferFunction()
        if len(groupSelected) > 0:
            groupSelectedList = list(groupSelected)
            rangeColorTransfer = [groupSelectedList[0], groupSelectedList[len(groupSelectedList) - 1]]
            colorTransferFunction.AdjustRange(rangeColorTransfer)
            for key, value in self.colorBar.items():
                # postion on the current arrow
                x = (groupSelectedList[len(groupSelectedList) - 1] - groupSelectedList[0]) * value[0] + groupSelectedList[0]
                # color of the current arrow
                r = value[1]
                g = value[2]
                b = value[3]
                colorTransferFunction.AddRGBPoint(x,r,g,b)
        return colorTransferFunction

    def deleteArrays(self, key, value):
        """ Function to copy and delete all the arrays of all the meshes contained in a list
        """
        for vtkFile in value:
            # Read VTK File
            reader = vtk.vtkDataSetReader()
            reader.SetFileName(vtkFile)
            reader.ReadAllVectorsOn()
            reader.ReadAllScalarsOn()
            reader.Update()
            polyData = reader.GetOutput()

            # Copy of the polydata
            polyDataCopy = vtk.vtkPolyData()
            polyDataCopy.DeepCopy(polyData)
            pointData = polyDataCopy.GetPointData()

            # Remove all the arrays
            numAttributes = pointData.GetNumberOfArrays()
            for i in range(0, numAttributes):
                pointData.RemoveArray(0)

            # Creation of the path of the vtk file without arrays to save it in the temporary directory of Slicer
            filename = os.path.basename(vtkFile)
            filepath = slicer.app.temporaryPath + '/' + filename

            # Save the vtk file without array in the temporary directory in Slicer
            self.saveVTKFile(polyDataCopy, filepath)
        return

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

    def printStatus(self, caller, event):
        print("Got a %s from a %s" % (event, caller.GetClassName()))
        if caller.IsA('vtkMRMLCommandLineModuleNode'):
            print("Status is %s" % caller.GetStatusString())
            # print("output:   \n %s" % caller.GetOutputText())
            # print("error:   \n %s" % caller.GetErrorText())
        return

    def computeMean(self, numGroup, vtkList):
        """ Function to compute the mean between all 
        the mesh-files contained in one group
        """
        print("--- Compute the mean of all the group ---")
        # Call of computeMean : computation of an average shape for a group of shaoes 
        # Arguments:
        #  --inputList is the list of vtkfile we want to compute the average
        #  --outputSurface is the resulting mean shape
        
        parameters = {}

        vtkfilelist = []
        for vtkFiles in vtkList:
            vtkfilelist.append(vtkFiles)
        parameters["inputList"] = vtkfilelist

        outModel = slicer.vtkMRMLModelNode()
        slicer.mrmlScene.AddNode(outModel)
        parameters["outputSurface"] = outModel.GetID()

        computeMean = slicer.modules.computemean
        
        # print parameters

        cliNode = slicer.cli.run(computeMean, None, parameters, wait_for_completion=True)
        cliNode.AddObserver('ModifiedEvent', self.printStatus)

        resultdir = slicer.app.temporaryPath
        slicer.util.saveNode(outModel, str(os.path.join(resultdir,"meanGroup" + str(numGroup) + ".vtk")))
        slicer.mrmlScene.RemoveNode(outModel) 

        return 
        


    def removeDataVTKFiles(self, value):
        """ Function to remove in the temporary directory all 
        the data used to create the mean for each group
        """
        # remove of all the vtk file
        for vtkFile in value:
            filepath = slicer.app.temporaryPath + '/' + os.path.basename(vtkFile)
            if os.path.exists(filepath):
                os.remove(filepath)

    # Function to storage the mean of each group in a dictionary
    def storageMean(self, dictGroups, key):
        filename = "meanGroup" + str(key)
        meanPath = slicer.app.temporaryPath + '/' + filename + '.vtk'
        dictGroups[key] = meanPath
        # print(dictGroups)

    def creationCSVFile(self, directory, CSVbasename, dictForCSV, option):
        """ Function to create a CSV file:
            - Two columns are always created:
                - First column: path of the vtk files
                - Second column: group associated to this vtk file
            - If saveH5 is True, this CSV file will contain a New Classification Group, a thrid column is then added
                - Thrid column: path of the shape model of each group
        """
        CSVFilePath = str(directory) + "/" + CSVbasename
        file = open(CSVFilePath, 'w')
        cw = csv.writer(file, delimiter=',')
        if option == "Groups":
            cw.writerow(['VTK Files', 'Group'])
        elif option == "MeanGroup":
            cw.writerow(['Mean shapes VTK Files', 'Group'])
        for key, value in dictForCSV.items():
            if type(value) is ListType:
                for vtkFile in value:
                    if option == "Groups":
                        cw.writerow([vtkFile, str(key)])
                
            elif option == "MeanGroup":
                cw.writerow([value, str(key)])

            elif option == "NCG":
                cw.writerow([value, str(key)])
        file.close()

    def saveNewClassificationGroups(self, basename, directory, dictShapeModels):

        """ Function to save the data of the new Classification Groups in the directory given by the user
            - The mean vtk files of each groups
            - The shape models of each groups
            - The CSV file containing:
                - First column: the paths of mean vtk file of each group
                - Second column: the groups associated
                - Third column: the paths of the shape model of each group
        """ 
        dictForCSV = dict()
        for key, value in dictShapeModels.items():
            # Save the shape model (h5 file) of each group
            h5Basename = "G" + str(key) + ".h5"
            oldh5path = slicer.app.temporaryPath + "/" + h5Basename
            newh5path = directory + "/" + h5Basename
            shutil.copyfile(oldh5path, newh5path)
            dictForCSV[key] = newh5path

        # Save the CSV file containing all the data useful in order to compute OAIndex of a patient
        self.creationCSVFile(directory, basename, dictForCSV, "NCG")

    def removeDataAfterNCG(self, dict):
        """ Function to remove in the temporary directory all the 
        data useless after to do a export of the new Classification Groups
        """
        for key in dict.keys():
            # Remove of the shape model of each group
            path = dict[key]
            if os.path.exists(path):
                os.remove(path)

    def actionOnDictionary(self, dict, file, listSaveVTKFiles, action):
        """ Function to make some action on a dictionary
            Action Remove:
                Remove the vtk file to the dictionary dict
                If the vtk file was found:
                    Return a list containing the key and the vtk file
                Else:
                    Return False
            Action Find:
                Find the vtk file in the dictionary dict
                If the vtk file was found:
                    Return True
                Else:
                    Return False
        """
        if action == 'remove' or action == 'find':
            if not file == None:
                for key, value in dict.items():
                    for vtkFile in value:
                        filename = os.path.basename(vtkFile)
                        if filename == file:
                            if action == 'remove':
                                value.remove(vtkFile)
                                listSaveVTKFiles.append(key)
                                listSaveVTKFiles.append(vtkFile)
                                return listSaveVTKFiles
                            return True
            return False

        # Action Add:
        #      Add a vtk file to the dictionary dict at the given key contained in the first case of the list
        if action == 'add':
            if not listSaveVTKFiles == None and not file == None:
                value = dict.get(listSaveVTKFiles[0], None)
                value.append(listSaveVTKFiles[1])


    def checkNumberOfPoints(self, dictShapes):
        """ Function to check that all the shapes have 
        the same number of points 
        """
        num_shape = 0
        num_points = 0 
        for key, value in dictShapes.items():
            if type(value) is ListType:
                for shape in value:
                    try:
                        reader_poly = vtk.vtkPolyDataReader()
                        reader_poly.SetFileName(shape)
                        reader_poly.Update()
                        geometry = reader_poly.GetOutput()
                        
                        if num_shape == 0:
                            num_points = geometry.GetNumberOfPoints()
                        else:
                            if not geometry.GetNumberOfPoints() == num_points:
                                slicer.util.errorDisplay('All the shapes must have the same number of points!')
                                return False
                                # raise Exception('Unexpected number of points in the shape: %s' % str(geometry.GetNumberOfPoints()))
                    
                    except IOError as e:
                        print('Could not read:', shape, ':', e, '- it\'s ok, skipping.')
                    
                num_shape = num_shape + 1
            else: 
                shape = value
                try:
                    reader_poly = vtk.vtkPolyDataReader()
                    reader_poly.SetFileName(shape)
                    reader_poly.Update()
                    geometry = reader_poly.GetOutput()
                    
                    if num_shape == 0:
                        num_points = geometry.GetNumberOfPoints()
                    else:
                        if not geometry.GetNumberOfPoints() == num_points:
                            slicer.util.errorDisplay('All the shapes must have the same number of points!')
                            return False
                            # raise Exception('Unexpected number of points in the shape: %s' % str(geometry.GetNumberOfPoints()))
                except IOError as e:
                    print('Could not read:', shape, ':', e, '- it\'s ok, skipping.')

        return num_points


    def extractFeatures(self, shape, meansList, outputDir, train):
        """ Function to extract the features from the provided shape
        Call the CLI surfacefeaturesextractor
        """
        # #     Creation of the command line
        # scriptedModulesPath = eval('slicer.modules.%s.path' % self.interface.moduleName.lower())
        # scriptedModulesPath = os.path.dirname(scriptedModulesPath)
        # libPath = os.path.join(scriptedModulesPath)
        # sys.path.insert(0, libPath)
        # if sys.platform == 'win32':
        #     surfacefeaturesextractor = os.path.join(scriptedModulesPath, '..', 'cli-modules', 'surfacefeaturesextractor.exe')
        # else:
        #     surfacefeaturesextractor = os.path.join(scriptedModulesPath, '..', 'cli-modules', 'surfacefeaturesextractor')            
        # # surfacefeaturesextractor = "/Users/mirclem/Desktop/work/ShapeVariationAnalyzer/src/CLI/SurfaceFeaturesExtractor-build/src/SurfaceFeaturesExtractor/bin/surfacefeaturesextractor"
        # # surfacefeaturesextractor = "/Users/prisgdd/Documents/Projects/CNN/SurfaceFeaturesExtractor-build/src/SurfaceFeaturesExtractor/bin/surfacefeaturesextractor"
        
        # filename = str(os.path.basename(shape))

        # arguments = list()

        # # Input Mesh
        # arguments.append(shape)

        # # Output Mesh
        # arguments.append(str(os.path.join(outputDir,filename))) 

        # # List of average shapes
        # arguments.append("--distMeshOn")
        # arguments.append("--distMesh")
        # arguments.append(str(meansList))
        # # print(arguments)

        # #     Call the executable
        # process = qt.QProcess()
        # process.setProcessChannelMode(qt.QProcess.MergedChannels)

        # # print("Calling " + os.path.basename(computeMean))
        # process.start(surfacefeaturesextractor, arguments)
        # process.waitForStarted()
        # # print("state: " + str(process.state()))
        # process.waitForFinished()
        # # print("error: " + str(process.error()))
        
        # processOutput = str(process.readAll())
        # print(processOutput)

        parameters = {}

        slicer.util.loadModel(shape)
        modelNode = slicer.util.getNode(os.path.basename(shape).split('.')[0])
        parameters["inputMesh"] = modelNode.GetID()

        filename = str(os.path.basename(shape))
        outModel = slicer.vtkMRMLModelNode()
        outModel.SetName(filename)
        slicer.mrmlScene.AddNode(outModel)
        parameters["outputMesh"] = outModel.GetID()

        parameters["distMeshOn"] = True
        parameters["distMesh"] = str(meansList)

        # print str(meansList)

        surfacefeaturesextractor = slicer.modules.surfacefeaturesextractor
        cliNode = slicer.cli.run(surfacefeaturesextractor, None, parameters, wait_for_completion=True)
        # cliNode = slicer.cli.runSync(surfacefeaturesextractor, None, parameters)
        cliNode.AddObserver('ModifiedEvent', self.printStatus)
        slicer.util.saveNode(outModel, str(os.path.join(outputDir,filename)))
        if train == True:
            slicer.mrmlScene.RemoveNode(modelNode)
            slicer.mrmlScene.RemoveNode(outModel)
        return 


    def storageFeaturesData(self, dictFeatData, dictShapeModels):
        """ Function to complete a dict listing all
        the shapes with extracted features
        """
        for key, value in dictShapeModels.items():
            newValue = list()
            for shape in value:
                filename,_ = os.path.splitext(os.path.basename(shape))
                ftPath = os.path.join(slicer.app.temporaryPath,'dataFeatures',filename + '.vtk')

                newValue.append(ftPath)
            dictFeatData[key] = newValue
        return

    def storageDataToClassify(self, dictFeatData, listPatient, outputDir):
        """ Funtion to complete a dict listing all 
        the shapes to be classified
        """
        for i in range(0, len(listPatient)):
            filename,_ = os.path.splitext(os.path.basename(listPatient[i]))
            ftPath = os.path.join(outputDir, filename + '.vtk')
            listPatient[i] = ftPath
        return listPatient

    def pickleData(self, dictFeatData, featuresList, controlGroup):
        """ Function to pickle the data for the network
        Update the inputData instance for update in the zipfile later
        """
        tempPath = slicer.app.temporaryPath
        networkDir = os.path.join(tempPath, "Network")
        if os.path.isdir(networkDir):
            shutil.rmtree(networkDir)
        os.mkdir(networkDir) 

        nbGroups = len(dictFeatData.keys())
        self.input_Data = inputData.inputData()
        self.input_Data.NUM_CLASSES = nbGroups

        nb_feat = len(featuresList)
        if featuresList.count('Normals'): 
            nb_feat += 2 
        if featuresList.count('Distances to average shapes'):
            nb_feat = nb_feat + nbGroups - 1
        if featuresList.count('Position'):
            nb_feat += 2

        self.input_Data.featuresList = featuresList
        self.input_Data.controlAverage = controlGroup
        self.input_Data.NUM_FEATURES = nb_feat

        reader_poly = vtk.vtkPolyDataReader()
        reader_poly.SetFileName(dictFeatData[0][0])

        reader_poly.Update()
        self.input_Data.NUM_POINTS = reader_poly.GetOutput().GetNumberOfPoints()

        for file in os.listdir(tempPath):
            if os.path.splitext(os.path.basename(file))[1] == '.pickle':
                os.remove(os.path.join(tempPath,file))

        dataset_names = self.input_Data.maybe_pickle(dictFeatData, 3, path=tempPath, force=False)

        # Save info in JSON File
        network_param = dict()
        network_param["NUM_CLASSES"] = self.input_Data.NUM_CLASSES
        network_param["NUM_FEATURES"] = self.input_Data.NUM_FEATURES
        network_param["NUM_POINTS"] = self.input_Data.NUM_POINTS
        network_param["Features"] = featuresList
        network_param["controlAverage"] = self.input_Data.controlAverage 

        jsonDict = dict()
        jsonDict["CondylesClassifier"] = network_param

        with open(os.path.join(networkDir,'classifierInfo.json'), 'w') as f:
            json.dump(jsonDict, f, ensure_ascii=False, indent = 4)

        #
        # Determine dataset size
        #
        small_classe = 100000000
        completeDataset = 0
        for key, value in dictFeatData.items():
            if len(value) < small_classe:
                small_classe = len(value)
            completeDataset = completeDataset + len(value)

        if small_classe < 4: 
            train_size = ( small_classe - 1 ) * nbGroups
        else: 
            train_size = ( small_classe - 3 ) * nbGroups
        valid_size = 3 * nbGroups
        test_size = completeDataset

        valid_dataset, valid_labels, train_dataset, train_labels = self.input_Data.merge_datasets(dataset_names, train_size, valid_size) 
        _, _, test_dataset, test_labels = self.input_Data.merge_all_datasets(dataset_names, test_size)

        train_dataset, train_labels = self.input_Data.randomize(train_dataset, train_labels)
        valid_dataset, valid_labels = self.input_Data.randomize(valid_dataset, valid_labels)
        test_dataset, test_labels = self.input_Data.randomize(test_dataset, test_labels)

        pickle_file = os.path.join(networkDir,'datasets.pickle')

        try:
            f = open(pickle_file, 'wb')
            save = {
                'train_dataset': train_dataset,
                'train_labels': train_labels,
                'valid_dataset': valid_dataset,
                'valid_labels': valid_labels,
                'test_dataset': test_dataset,
                'test_labels': test_labels,
            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

        statinfo = os.stat(pickle_file)
        print('Compressed pickle size:', statinfo.st_size)

        return pickle_file


    def pickleToClassify(self, patientList, path):
        """ Function to pickle the data to classify for the network
        """
        force = True

        set_filename = os.path.join(path, 'toClassify.pickle')
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset, allShapes_feat = self.input_Data.load_features_with_names(patientList)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(allShapes_feat, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

        return set_filename

    def exportUntrainedNetwork(self, archiveName, zipPath, num_steps, num_layers):
        """ Funciton to compress/zip everything needed to 
        export the Classifier BEFORE training 
        Useful for remote training 
        """
        # Set path for teh network
        tempPath = slicer.app.temporaryPath
        networkDir = os.path.join(tempPath, "Network")
        if os.path.isdir(networkDir):
            shutil.rmtree(networkDir)
        os.mkdir(networkDir) 

        with zipfile.ZipFile(archiveName) as zf:
            zf.extractall(os.path.dirname(archiveName))
            
        jsonFile = os.path.join(networkDir, 'classifierInfo.json')

        with open(jsonFile) as f:
            jsonDict = json.load(f)

        jsonDict['CondylesClassifier']['learning_rate'] = 0.0005
        jsonDict['CondylesClassifier']['lambda_reg'] = 0.01
        jsonDict['CondylesClassifier']['num_epochs'] = 5
        jsonDict['CondylesClassifier']['num_steps'] =  num_steps
        jsonDict['CondylesClassifier']['batch_size'] = 10
        jsonDict['CondylesClassifier']['NUM_HIDDEN_LAYERS'] = num_layers
        
        if jsonDict['CondylesClassifier']['NUM_HIDDEN_LAYERS'] == 1:
            jsonDict['CondylesClassifier']['nb_hidden_nodes_1'] = int ( jsonDict['CondylesClassifier']['NUM_POINTS'] * jsonDict['CondylesClassifier']['NUM_FEATURES'] + jsonDict['CondylesClassifier']['NUM_CLASSES'] // 2 )
            # jsonDict['CondylesClassifier']['nb_hidden_nodes_1'] = int ( math.sqrt ( jsonDict['CondylesClassifier']['NUM_POINTS'] * jsonDict['CondylesClassifier']['NUM_FEATURES'] * jsonDict['CondylesClassifier']['NUM_CLASSES'] ))
            jsonDict['CondylesClassifier']['nb_hidden_nodes_2'] = 0
        
        elif jsonDict['CondylesClassifier']['NUM_HIDDEN_LAYERS'] == 2:
            r = math.pow( jsonDict['CondylesClassifier']['NUM_POINTS'] * jsonDict['CondylesClassifier']['NUM_FEATURES'] / jsonDict['CondylesClassifier']['NUM_CLASSES'], 1/3)
            jsonDict['CondylesClassifier']['nb_hidden_nodes_1'] = int ( jsonDict['CondylesClassifier']['NUM_CLASSES'] * math.pow ( r, 2 ))
            jsonDict['CondylesClassifier']['nb_hidden_nodes_2'] =int ( jsonDict['CondylesClassifier']['NUM_POINTS'] * jsonDict['CondylesClassifier']['NUM_FEATURES'] * r )

        with open(os.path.join(networkDir,'classifierInfo.json'), 'w') as f:
            json.dump(jsonDict, f, ensure_ascii=False, indent = 4)

        path = zipPath.split(".zip",1)[0]
        shutil.make_archive(path, 'zip', networkDir)

        return 


    def trainNetworkClassification(self, archiveName, num_steps, num_layers):
        """ Function to train the Neural Network
        within the virtualenv containing tensorflow
        First creation of a zipfile with updated info
        Return the estimated accuracy of the network
        """
        # Set path for teh network
        tempPath = slicer.app.temporaryPath
        networkDir = os.path.join(tempPath, "Network")
        if os.path.isdir(networkDir):
            shutil.rmtree(networkDir)
        os.mkdir(networkDir) 

        with zipfile.ZipFile(archiveName) as zf:
            zf.extractall(os.path.dirname(archiveName))

        jsonFile = os.path.join(networkDir, 'classifierInfo.json')

        with open(jsonFile) as f:
            jsonDict = json.load(f)

        jsonDict['CondylesClassifier']['learning_rate'] = 0.0005
        jsonDict['CondylesClassifier']['lambda_reg'] = 0.01
        jsonDict['CondylesClassifier']['num_epochs'] = 50
        jsonDict['CondylesClassifier']['num_steps'] =  num_steps
        # jsonDict['CondylesClassifier']['num_steps'] =  2001
        jsonDict['CondylesClassifier']['batch_size'] = 10
        jsonDict['CondylesClassifier']['NUM_HIDDEN_LAYERS'] = num_layers
        
        if jsonDict['CondylesClassifier']['NUM_HIDDEN_LAYERS'] == 1:
            jsonDict['CondylesClassifier']['nb_hidden_nodes_1'] = int ( math.sqrt ( jsonDict['CondylesClassifier']['NUM_POINTS'] * jsonDict['CondylesClassifier']['NUM_FEATURES'] * jsonDict['CondylesClassifier']['NUM_CLASSES'] ))
            jsonDict['CondylesClassifier']['nb_hidden_nodes_2'] = 0
        
        elif jsonDict['CondylesClassifier']['NUM_HIDDEN_LAYERS'] == 2:
            r = math.pow( jsonDict['CondylesClassifier']['NUM_POINTS'] * jsonDict['CondylesClassifier']['NUM_FEATURES'] / jsonDict['CondylesClassifier']['NUM_CLASSES'], 1/3)
            jsonDict['CondylesClassifier']['nb_hidden_nodes_1'] = int ( jsonDict['CondylesClassifier']['NUM_CLASSES'] * math.pow ( r, 2 ))
            jsonDict['CondylesClassifier']['nb_hidden_nodes_2'] =int ( jsonDict['CondylesClassifier']['NUM_POINTS'] * jsonDict['CondylesClassifier']['NUM_FEATURES'] * r )

        with open(os.path.join(networkDir,'classifierInfo.json'), 'w') as f:
            json.dump(jsonDict, f, ensure_ascii=False, indent = 4)

        shutil.make_archive(base_name = networkDir, format = 'zip', root_dir = tempPath, base_dir = 'Network')

        # 
        # Train network in virtualenv
        # 
        currentPath = os.path.dirname(os.path.abspath(__file__))
        train_file = os.path.join(currentPath,'Resources','Classifier','trainNeuralNetwork.py')
        envWrapper_file = os.path.join(currentPath,'Wrapper','envTensorFlowWrapper.py')

        if slicer.app.isInstalled:
            pathSlicerExec = str(os.path.dirname(sys.executable))
            pathSlicerPython = os.path.join(pathSlicerExec, "..", "bin", "SlicerPython")
        else:
            pathSlicerPython = os.path.join(os.environ["SLICER_HOME"], "..", "python-install", "bin", "SlicerPython")

        args = '{"--inputZip": "' + archiveName + '", "--outputZip": "' + archiveName + '"}' 
        command = [pathSlicerPython, envWrapper_file, "-pgm", train_file, "-args", args ]

        print command
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err =  p.communicate()
        print("\nout : " + str(out) + "\nerr : " + str(err))

        if os.path.isdir(networkDir):
            shutil.rmtree(networkDir)
        os.mkdir(networkDir) 

        with zipfile.ZipFile(archiveName) as zf:
            zf.extractall(os.path.dirname(archiveName))

        jsonFile = os.path.join(networkDir, 'classifierInfo.json')
        with open(jsonFile) as f:
            jsonDict = json.load(f)

        return jsonDict['CondylesClassifier']['accuracy']


    def zipdir(self, path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file))

    def exportModelNetwork(self, filepath):
        tempPath = slicer.app.temporaryPath
        networkDir = os.path.join(tempPath, 'Network')

        # Zipper tout ca 
        shutil.make_archive(base_name = filepath, format = 'zip', root_dir = tempPath, base_dir = 'Network')

        return

    def validModelNetwork(self, archiveName, dictGroups):
        """ Function to valid a given neural network 
        when its path is specified 
        If it is accepted, parameters are updated for later reuse
        """
        tempPath = slicer.app.temporaryPath
        networkDir = os.path.join(tempPath, 'Network')
        # Si y a deja un Network dans le coin, on le degage
        if os.path.isdir(networkDir):
            shutil.rmtree(networkDir)
        os.mkdir(networkDir) 

        # Unpack archive
        with zipfile.ZipFile(archiveName) as zf:
            zf.extractall(tempPath)

        # check JSON file
        jsonFile = os.path.join(networkDir, 'classifierInfo.json')
        saveModelPath = os.path.join(networkDir, 'CondylesClassifier')

        with open(jsonFile) as f:    
            jsonDict = json.load(f)

        # In case our JSON file doesnt contain a valid Classifier
        if not jsonDict.has_key('CondylesClassifier'):
            print("No CondylesClassifier containing network parameters")
            return 0
        myDict = jsonDict['CondylesClassifier']
        if not ('NUM_CLASSES' in myDict and 'NUM_FEATURES' in myDict and 'NUM_POINTS' in myDict):
            print("Missing basics network parameters")
            return 0

        self.input_Data = inputData.inputData()
        self.input_Data.NUM_POINTS = jsonDict['CondylesClassifier']['NUM_POINTS']
        self.input_Data.NUM_CLASSES = jsonDict['CondylesClassifier']['NUM_CLASSES']
        self.input_Data.NUM_FEATURES = jsonDict['CondylesClassifier']['NUM_FEATURES']
        self.input_Data.featuresList = jsonDict['CondylesClassifier']['Features']
        self.input_Data.controlAverage  = jsonDict['CondylesClassifier']['controlAverage'] 

        numModelFiles = 0
        strCondClass = 'CondylesClassifier'
        for f in os.listdir(networkDir):
            if f[0:len('CondylesClassifier')] == 'CondylesClassifier' : 
                numModelFiles = numModelFiles + 1

        if numModelFiles < 3 :
            print("Missing files for this model")
            return 0

        # Get the dict of mean shapes
        meanGroupsDir = os.path.join(networkDir, 'meanGroups')
        jsonMeans = os.path.join(meanGroupsDir, 'meanGroups.json')
        with open(jsonMeans) as f:    
            meanGroupsDict = json.load(f) 
            for group, file in meanGroupsDict.items():
                dictGroups[group] = os.path.join(meanGroupsDir,file)

        return 1

    # Function to compute the OA index of a patient
    def evalClassification(self, archiveName):
        # Set le path pour le network
        tempPath = slicer.app.temporaryPath
        networkDir = os.path.join(tempPath, "Network")
        if os.path.isdir(networkDir):
            shutil.rmtree(networkDir)
        os.mkdir(networkDir) 

        # Classify dans le virtualenv
        currentPath = os.path.dirname(os.path.abspath(__file__))
        train_file = os.path.join(currentPath,'Resources','Classifier','evalShape.py')
        envWrapper_file = os.path.join(currentPath,'Wrapper','envTensorFlowWrapper.py')
        
        pathSlicerExec = str(os.path.dirname(sys.executable))
        pathSlicerPython = os.path.join(pathSlicerExec, "..", "bin", "SlicerPython")
        
        args = '{"--inputZip": "' + archiveName + '", "--outputZip": "' + archiveName + '"}' 
        command = [pathSlicerPython, envWrapper_file, "-pgm", train_file, "-args", args ]

        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err =  p.communicate()
        print("\nout : " + str(out) + "\nerr : " + str(err))

        with zipfile.ZipFile(archiveName) as zf:
            zf.extractall(os.path.dirname(archiveName))

        jsonFile = os.path.join(networkDir, 'results.json')
        with open(jsonFile) as f:
            resultsDict = json.load(f)

        # print(str(resultsDict))
        return resultsDict


    def creationCSVFileForResult(self, table, directory, CSVbasename):
        """ Function to store all the results in a CSV file
        """
        CSVFilePath = directory + "/" + CSVbasename
        file = open(CSVFilePath, 'w')
        cw = csv.writer(file, delimiter=',')
        cw.writerow(['VTK Files', 'Assigned Group'])
        for row in range(0,table.rowCount):
            # Recovery of the filename of vtk file
            qlabel = table.cellWidget(row, 0)
            vtkFile = qlabel.text
            # Recovery of the assigned group
            qlabel = table.cellWidget(row, 1)
            assignedGroup = qlabel.text

            # Write the result in the CSV File
            cw.writerow([vtkFile, str(assignedGroup)])

class ShapeVariationAnalyzerTest(ScriptedLoadableModuleTest):
    pass
