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
import neuralNetwork as nn
import numpy as np
import tensorflow as tf
import zipfile


class Classification(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Classification"
        parent.categories = ["Quantification"]
        parent.dependencies = []
        parent.contributors = ["Priscille de Dumast (University of Michigan)"]
        parent.helpText = """
            Classification is used to define the OA type of
            a patient according a Classification Groups that
            you can create.
            """
        parent.acknowledgementText = """
            This work was supported by the National
            Institutes of Dental and Craniofacial Research
            and Biomedical Imaging and Bioengineering of
            the National Institutes of Health under Award
            Number R01DE024450.
            """


class ClassificationWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # ---- Widget Setup ----

        # Global Variables
        self.logic = ClassificationLogic(self)
        self.dictVTKFiles = dict()
        self.dictGroups = dict()
        self.dictCSVFile = dict()
        self.directoryList = list()
        self.groupSelected = set()
        self.dictShapeModels = dict()
        self.patientList = list()
        self.dictResult = dict()

        self.dictFeatData = dict()

        # Interface
        loader = qt.QUiLoader()
        self.moduleName = 'Classification'
        scriptedModulesPath = eval('slicer.modules.%s.path' % self.moduleName.lower())
        scriptedModulesPath = os.path.dirname(scriptedModulesPath)
        path = os.path.join(scriptedModulesPath, 'Resources', 'UI', '%s.ui' % self.moduleName)
        qfile = qt.QFile(path)
        qfile.open(qt.QFile.ReadOnly)

        widget = loader.load(qfile, self.parent)
        self.layout = self.parent.layout()
        self.widget = widget
        self.layout.addWidget(widget)

        #     global variables of the Interface:
        #          Tab: Creation of CSV File for Classification Groups
        self.collapsibleButton_creationCSVFile = self.logic.get('CollapsibleButton_creationCSVFile')
        self.spinBox_group = self.logic.get('spinBox_group')
        self.directoryButton_creationCSVFile = self.logic.get('DirectoryButton_creationCSVFile')
        self.stackedWidget_manageGroup = self.logic.get('stackedWidget_manageGroup')
        self.pushButton_addGroup = self.logic.get('pushButton_addGroup')
        self.pushButton_removeGroup = self.logic.get('pushButton_removeGroup')
        self.pushButton_modifyGroup = self.logic.get('pushButton_modifyGroup')
        self.directoryButton_exportCSVFile = self.logic.get('DirectoryButton_exportCSVFile')
        self.pushButton_exportCSVfile = self.logic.get('pushButton_exportCSVfile')
        #          Tab: Creation of New Classification Groups
        self.collapsibleButton_previewClassificationGroups = self.logic.get('CollapsibleButton_previewClassificationGroups')
        self.pathLineEdit_previewGroups = self.logic.get('pathLineEdit_previewGroups')
        self.collapsibleGroupBox_previewVTKFiles = self.logic.get('CollapsibleGroupBox_previewVTKFiles')
        self.checkableComboBox_ChoiceOfGroup = self.logic.get('CheckableComboBox_ChoiceOfGroup')
        self.tableWidget_VTKFiles = self.logic.get('tableWidget_VTKFiles')
        self.pushButton_previewVTKFiles = self.logic.get('pushButton_previewVTKFiles')
        # self.pushButton_compute = self.logic.get('pushButton_compute')
        self.directoryButton_exportUpdatedClassification = self.logic.get('DirectoryButton_exportUpdatedClassification')
        self.pushButton_exportUpdatedClassification = self.logic.get('pushButton_exportUpdatedClassification')
                 # Tab: Selection Classification Groups
        # self.collapsibleButton_SelectClassificationGroups = self.logic.get('CollapsibleButton_SelectClassificationGroups')
        # self.pathLineEdit_selectionClassificationGroups = self.logic.get('PathLineEdit_selectionClassificationGroups')
        self.comboBox_healthyGroup = self.logic.get('comboBox_healthyGroup')
        # self.pushButton_previewGroups = self.logic.get('pushButton_previewGroups')
        # self.MRMLTreeView_classificationGroups = self.logic.get('MRMLTreeView_classificationGroups')
        #          Tab: Select Input Data
        self.collapsibleButton_classificationNetwork = self.logic.get('collapsibleButton_classificationNetwork')
        self.MRMLNodeComboBox_VTKInputData = self.logic.get('MRMLNodeComboBox_VTKInputData')
        self.pathLineEdit_CSVInputData = self.logic.get('PathLineEdit_CSVInputData')
        # self.checkBox_fileInGroups = self.logic.get('checkBox_fileInGroups')
        self.pushButton_classifyIndex = self.logic.get('pushButton_classifyIndex')

        self.pushButton_trainNetwork = self.logic.get('pushButton_trainNetwork')
        self.pushButton_exportNetwork = self.logic.get('pushButton_ExportNetwork')
        self.pathLineEdit_CSVFileDataset = self.logic.get('pathLineEdit_CSVFileDataset')
        self.pathLineEdit_CSVFileMeansShape = self.logic.get('pathLineEdit_CSVFileMeansShape')
        self.directoryButton_exportNetwork = self.logic.get('directoryButton_ExportNetwork')
        self.pushButton_preprocessData = self.logic.get('pushButton_preprocessData')
        self.label_trainNetwork = self.logic.get('label_trainNetwork')

        self.pathLineEdit_networkPath = self.logic.get('ctkPathLineEdit_networkPath')
        self.pathLineEdit_CSVFileMeansShapeClassify = self.logic.get('pathLineEdit_CSVFileMeansShapeClassify')


        #          Tab: Result / Analysis
        self.collapsibleButton_Result = self.logic.get('CollapsibleButton_Result')
        self.tableWidget_result = self.logic.get('tableWidget_result')
        self.pushButton_exportResult = self.logic.get('pushButton_exportResult')
        self.directoryButton_exportResult = self.logic.get('DirectoryButton_exportResult')

        
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
        # self.pushButton_previewGroups.setDisabled(True)
        # self.pushButton_compute.setDisabled(True)
        # self.pushButton_compute.setDisabled(True)
        self.directoryButton_exportUpdatedClassification.setDisabled(True)
        self.pushButton_exportUpdatedClassification.setDisabled(True)
        # self.checkBox_fileInGroups.setDisabled(True)
        # self.checkableComboBox_ChoiceOfGroup.setDisabled(True)
        self.tableWidget_VTKFiles.setDisabled(True)
        self.pushButton_previewVTKFiles.setDisabled(True)

        self.pushButton_previewGroups.setDisabled(True)
        self.pushButton_computeMeanGroup.setDisabled(True)
        self.directoryButton_exportMeanGroups.setDisabled(True)
        self.pushButton_exportMeanGroups.setDisabled(True)

        self.pushButton_trainNetwork.setDisabled(True)
        self.pushButton_exportNetwork.setDisabled(True)
        self.directoryButton_exportNetwork.setDisabled(True)
        self.pushButton_preprocessData.setDisabled(True)

        self.label_trainNetwork.hide()


        #     qMRMLNodeComboBox configuration
        self.MRMLNodeComboBox_VTKInputData.setMRMLScene(slicer.mrmlScene)

        #     initialisation of the stackedWidget to display the button "add group"
        self.stackedWidget_manageGroup.setCurrentIndex(0)

        #     spinbox configuration in the tab "Creation of CSV File for Classification Groups"
        self.spinBox_group.setMinimum(0)
        self.spinBox_group.setMaximum(0)
        self.spinBox_group.setValue(0)

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
        # self.tableWidget_VTKFiles.verticalHeader().setVisible(False)
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
        # self.pushButton_compute.connect('clicked()', self.onComputeNewClassificationGroups)
        self.pushButton_exportUpdatedClassification.connect('clicked()', self.onExportUpdatedClassificationGroups)
        # #          Tab: Selection of Classification Groups
        # self.collapsibleButton_SelectClassificationGroups.connect('clicked()',
        #                                                           lambda: self.onSelectedCollapsibleButtonOpen(self.collapsibleButton_SelectClassificationGroups))
        # self.pathLineEdit_selectionClassificationGroups.connect('currentPathChanged(const QString)', self.onSelectionClassificationGroups)
        # self.pushButton_previewGroups.connect('clicked()', self.onPreviewGroupMeans)
        #          Tab: Select Input Data
        self.collapsibleButton_classificationNetwork.connect('clicked()',
                                                       lambda: self.onSelectedCollapsibleButtonOpen(self.collapsibleButton_classificationNetwork))
        self.MRMLNodeComboBox_VTKInputData.connect('currentNodeChanged(vtkMRMLNode*)', self.onVTKInputData)
        # self.checkBox_fileInGroups.connect('clicked()', self.onCheckFileInGroups)
        self.pathLineEdit_CSVInputData.connect('currentPathChanged(const QString)', self.onCSVInputData)
        
        
        
        self.pushButton_classifyIndex.connect('clicked()', self.onClassifyIndex)
        #          Tab: Result / Analysis
        self.collapsibleButton_Result.connect('clicked()',
                                              lambda: self.onSelectedCollapsibleButtonOpen(self.collapsibleButton_Result))
        self.pushButton_exportResult.connect('clicked()', self.onExportResult)

        slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndCloseEvent, self.onCloseScene)


                 
        self.pathLineEdit_selectionClassificationGroups.connect('currentPathChanged(const QString)', self.onComputeAverageClassificationGroups)
        self.pushButton_previewGroups.connect('clicked()', self.onPreviewGroupMeans)
        self.pushButton_computeMeanGroup.connect('clicked()', self.onComputeMeanGroup)
        self.pushButton_exportMeanGroups.connect('clicked()', self.onExportMeanGroups)
        self.pathLineEdit_meanGroup.connect('currentPathChanged(const QString)', self.onMeanGroupCSV)

                # Tab: Classification Network
        self.pushButton_trainNetwork.connect('clicked()', self.onTrainNetwork)
        self.pushButton_exportNetwork.connect('clicked()', self.onExportNetwork)
        self.pathLineEdit_CSVFileDataset.connect('currentPathChanged(const QString)', self.onCSVFileDataset)
        self.pathLineEdit_CSVFileMeansShape.connect('currentPathChanged(const QString)', self.onCSVFileMeansShape)
        self.pushButton_preprocessData.connect('clicked()', self.onPreprocessData)
        self.stateCSVMeansShape = False
        self.stateCSVDataset = False
        self.pathLineEdit_networkPath.connect('currentPathChanged(const QString)', self.onNetworkPath)
        self.pathLineEdit_CSVFileMeansShapeClassify.connect('currentPathChanged(const QString)', self.onCSVFileMeansShapeClassify)

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

        # numItem = self.comboBox_healthyGroup.count
        # for i in range(0, numItem):
        #     self.comboBox_healthyGroup.removeItem(0)
            
        self.comboBox_healthyGroup.clear()

        print "onCloseScene"
        self.dictVTKFiles = dict()
        self.dictGroups = dict()
        self.dictCSVFile = dict()
        self.directoryList = list()
        self.groupSelected = set()
        self.dictShapeModels = dict()
        self.patientList = list()
        self.dictResult = dict()

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
        self.directoryButton_exportUpdatedClassification.setDisabled(True)
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

    # Only one tab can be display at the same time:
    #   When one tab is opened all the other tabs are closed
    def onSelectedCollapsibleButtonOpen(self, selectedCollapsibleButton):
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

    # Function in order to manage the display of these three buttons:
    #    - "Add Group"
    #    - "Modify Group"
    #    - "Remove Group"
    def onManageGroup(self):
        # Display the button:
        #     - "Add Group" for a group which hasn't been added yet
        #     - "Remove Group" for the last group added
        #     - "Modify Group" for all the groups added
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

    # Function to add a group of the dictionary
    #    - Add the paths of all the vtk files found in the directory given
    #      of a dictionary which will be used to create the CSV file
    def onAddGroupForCreationCSVFile(self):
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
        # print self.dictCSVFile

    # Function to remove a group of the dictionary
    #    - Remove the paths of all the vtk files corresponding to the selected group
    #      of the dictionary which will be used to create the CSV file
    def onRemoveGroupForCreationCSVFile(self):
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

    # Function to modify a group of the dictionary:
    #    - Remove of the dictionary the paths of all vtk files corresponding to the selected group
    #    - Add of the dictionary the new paths of all the vtk files
    def onModifyGroupForCreationCSVFile(self):
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

    # Function to export the CSV file in the directory chosen by the user
    #    - Save the CSV file from the dictionary previously filled
    #    - Load automatically this CSV file in the next tab: "Creation of New Classification Groups"
    def onExportForCreationCSVFile(self):
        # Path of the csv file
        directory = self.directoryButton_exportCSVFile.directory.encode('utf-8')
        basename = 'Groups.csv'
        filepath = directory + "/" + basename

        # Message if the csv file already exists
        messageBox = ctk.ctkMessageBox()
        messageBox.setWindowTitle(' /!\ WARNING /!\ ')
        messageBox.setIcon(messageBox.Warning)
        if os.path.exists(filepath):
            messageBox.setText('File ' + filepath + ' already exists!')
            messageBox.setInformativeText('Do you want to replace it ?')
            messageBox.setStandardButtons( messageBox.No | messageBox.Yes)
            choice = messageBox.exec_()
            if choice == messageBox.No:
                return

        # Save the CSV File
        self.logic.creationCSVFile(directory, basename, self.dictCSVFile, "Groups")

        # Re-Initialization of the first tab
        self.spinBox_group.setMaximum(1)
        self.spinBox_group.setValue(1)
        self.stackedWidget_manageGroup.setCurrentIndex(0)
        self.directoryButton_creationCSVFile.directory = qt.QDir.homePath() + '/Desktop'
        self.directoryButton_exportCSVFile.directory = qt.QDir.homePath() + '/Desktop'

        # Re-Initialization of:
        #     - the dictionary containing all the paths of the vtk groups
        #     - the list containing all the paths of the different directories
        self.directoryList = list()
        self.dictCSVFile = dict()

        # Message in the python console
        print "Export CSV File: " + filepath

        # Load automatically the CSV file in the pathline in the next tab "Creation of New Classification Groups"
        self.pathLineEdit_previewGroups.setCurrentPath(filepath)
        self.pathLineEdit_selectionClassificationGroups.setCurrentPath(filepath)

    # ---------------------------------------------------- #
    #     Tab: Creation of New Classification Groups       #
    #     
    #     Preview/Update classification Groups
    #     
    # ---------------------------------------------------- #

    # Function to read the CSV file containing all the vtk filepaths needed to create the new Classification Groups
    def onSelectPreviewGroups(self):
        # Re-initialization of the dictionary containing all the vtk files
        # which will be used to create a new Classification Groups
        self.dictVTKFiles = dict()

        # Check if the path exists:
        if not os.path.exists(self.pathLineEdit_previewGroups.currentPath):
            return

        print "------ Creation of a new Classification Groups ------"
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

    # Function to manage the checkable combobox to allow the user to choose the group that he wants to preview in SPV
    def onCheckableComboBoxValueChanged(self):
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

    # Function to manage the combobox which allow the user to change the group of a vtk file
    def onGroupValueChanged(self):
        # Updade the dictionary which containing the VTK files sorted by groups
        self.logic.onComboBoxTableValueChanged(self.dictVTKFiles, self.tableWidget_VTKFiles)

        # Update the checkable combobox which display the groups selected to preview them in SPV
        self.onCheckBoxTableValueChanged()

        # Enable exportation of the last updated csv file
        self.directoryButton_exportUpdatedClassification.setEnabled(True)
        self.pushButton_exportUpdatedClassification.setEnabled(True)
        # Default path to override the previous one
        self.directoryButton_exportUpdatedClassification.directory = os.path.dirname(self.pathLineEdit_previewGroups.currentPath)


    # Function to manage the checkbox in the table used to make a preview in SPV
    def onCheckBoxTableValueChanged(self):
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

    # Function to update the colors that the selected vtk files will have in Shape Population Viewer
    def updateColorInTableForPreviewInSPV(self, colorTransferFunction):
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

    # Function to display the selected vtk files in Shape Population Viewer
    #    - Add a color map "DisplayClassificationGroup"
    #    - Launch the CLI ShapePopulationViewer
    def onPreviewVTKFiles(self):
        print "--- Preview VTK Files in ShapePopulationViewer ---"
        if os.path.exists(self.pathLineEdit_previewGroups.currentPath):
            # Creation of a color map to visualize each group with a different color in ShapePopulationViewer
            self.logic.addColorMap(self.tableWidget_VTKFiles, self.dictVTKFiles)

            # Creation of a CSV file to load the vtk files in ShapePopulationViewer
            filePathCSV = slicer.app.temporaryPath + '/' + 'VTKFilesPreview_OAIndex.csv'
            self.logic.creationCSVFileForSPV(filePathCSV, self.tableWidget_VTKFiles, self.dictVTKFiles)

            # Launch the CLI ShapePopulationViewer
            parameters = {}
            parameters["CSVFile"] = filePathCSV
            launcherSPV = slicer.modules.launcher
            slicer.cli.run(launcherSPV, None, parameters, wait_for_completion=True)

            # Remove the vtk files previously created in the temporary directory of Slicer
            for value in self.dictVTKFiles.values():
                self.logic.removeDataVTKFiles(value)

    # Function to compute the new Classification Groups
    #    - Remove all the arrays of all the vtk files
    #    - Compute the mean of each group thanks to Statismo
    # def onComputeNewClassificationGroups(self):
    #     for key, value in self.dictVTKFiles.items():
    #         # Delete all the arrays in vtk file
    #         self.logic.deleteArrays(key, value)

    #         # Compute the shape model of each group
    #         self.logic.buildShapeModel(key, value)

    #         # Remove the vtk files used to create the shape model of each group
    #         self.logic.removeDataVTKFiles(value)

    #         # Storage of the shape model for each group
    #         self.logic.storeShapeModel(self.dictShapeModels, key)

    #     # Enable the option to export the new data
    #     self.directoryButton_exportUpdatedClassification.setEnabled(True)
    #     self.pushButton_exportUpdatedClassification.setEnabled(True)

    # Function to export the new Classification Groups
    #    - Data saved:
    #           - Save the mean vtk files in the selected directory
    #           - Save the CSV file in the selected directory
    #    - Load automatically the CSV file in the next tab: "Selection of Classification Groups"
    def onExportUpdatedClassificationGroups(self):
        print "--- Export the new Classification Groups ---"

        # Message for the user if files already exist
        directory = self.directoryButton_exportUpdatedClassification.directory.encode('utf-8')
        messageBox = ctk.ctkMessageBox()
        messageBox.setWindowTitle(' /!\ WARNING /!\ ')
        messageBox.setIcon(messageBox.Warning)
        filePathExisting = list()

        #   Check if the CSV file exists
        # CSVfilePath = directory + "/ClassificationGroups.csv"
        CSVfilePath = directory + "/Groups.csv"
        if os.path.exists(CSVfilePath):
            filePathExisting.append(CSVfilePath)

        #   Check if the shape model exist
        # print "dictshapemodels :: " + str(self.dictShapeModels)
        for key, value in self.dictVTKFiles.items():
            for shape in value:
                modelFilename = os.path.basename(shape)
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

        # Save the CSV File and the shape model of each group
        # self.logic.saveNewClassificationGroups('Groups.csv', directory, self.dictShapeModels)
        self.logic.creationCSVFile(directory, "Groups.csv", self.dictVTKFiles, "Groups")


        # Remove the shape model (GX.h5) of each group
        # self.logic.removeDataAfterNCG(self.dictVTKFiles)

        # Re-Initialization of the dictionary containing the path of the shape model of each group
        # self.dictVTKFiles = dict()

        # Message for the user
        slicer.util.delayDisplay("Files Saved")

        # Disable the option to export the new data
        self.directoryButton_exportUpdatedClassification.setDisabled(True)
        self.pushButton_exportUpdatedClassification.setDisabled(True)

        # Load automatically the CSV file in the pathline in the next tab "Selection of Classification Groups"
        if self.pathLineEdit_selectionClassificationGroups.currentPath == CSVfilePath:
            self.pathLineEdit_selectionClassificationGroups.setCurrentPath(" ")
        self.pathLineEdit_selectionClassificationGroups.setCurrentPath(CSVfilePath)

    # ---------------------------------------------------- #
    #        Tab: Selection of Classification Groups       #
    #        
    #        Compute Average groups!!
    #        
    # ---------------------------------------------------- #

    # Function to select the Classification Groups
    def onComputeAverageClassificationGroups(self):
        # Re-initialization of the dictionary containing the Classification Groups
        self.dictShapeModels = dict()

        # Check if the path exists:
        if not os.path.exists(self.pathLineEdit_selectionClassificationGroups.currentPath):
            return

        print "------ Selection of a Classification Groups ------"
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
        # Enable/disable buttons
        # self.comboBox_healthyGroup.setEnabled(True)
        

        self.pushButton_computeMeanGroup.setEnabled(True)

        # Configuration of the spinbox specify the healthy group
        #      Set the Maximum value of comboBox_healthyGroup at the maximum number groups


        # self.comboBox_healthyGroup.setMaximum(len(self.dictShapeModels) - 1)

    def onComputeMeanGroup(self):

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
        
        print "------ onMeanGroupCSV ------"
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
            # self.pathLineEdit_meanGroup.setCurrentPath(" ")
            return

        self.pushButton_previewGroups.setEnabled(True)
        self.comboBox_healthyGroup.setEnabled(True)
        # self.comboBox_healthyGroup.setMaximum(len(self.dictGroups.keys()) - 1)

        # numItem = self.comboBox_healthyGroup.count
        # for i in range(0, numItem):
        #     self.comboBox_healthyGroup.removeItem(0)
        self.comboBox_healthyGroup.clear()

        for key, value in self.dictGroups.items():
            # Fill the Checkable Combobox
            self.comboBox_healthyGroup.addItem("Group " + str(key))


        

    # Function to preview the Classification Groups in Slicer
    #    - The opacity of all the vtk files is set to 0.8
    #    - The healthy group is white and the others are red
    def onPreviewGroupMeans(self):
        print "------ Preview of the Group's Mean in Slicer ------"

        # for group, h5path in self.dictShapeModels.items():
        #     # Compute the mean of each group thanks to Statismo
        #     self.logic.computeMean(group, h5path)

        #     # Storage of the means for each group
        #     self.logic.storageMean(self.dictGroups, group)

        # If the user doesn't specify the healthy group
        #     error message for the user
        # Else
        #     load the Classification Groups in Slicer
        # if self.spinBox_healthyGroup.value == 0:
        #     # Error message:
        #     slicer.util.errorDisplay('Miss the number of the healthy group ')
        # else:

        # list = slicer.mrmlScene.GetNodesByClass("vtkMRMLModelNode")
        # end = list.GetNumberOfItems()
        # for i in range(0,end):
        #     model = list.GetItemAsObject(i)
        #     print model.GetName()
        #     print 
        #     if model.GetName()[:len("meanGroup")] == "meanGroup":
        #         hardenModel = slicer.mrmlScene.GetNodesByName(model.GetName()).GetItemAsObject(0)
        #         slicer.mrmlScene.RemoveNode(hardenModel)


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
        print "--- Export all the mean shapes + csv file ---"

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
        print "Saved in :: " + directory + "/MeanGroups.csv"

        # Disable the option to export the new data
        # self.directoryButton_exportUpdatedClassification.setDisabled(True)
        # self.pushButton_exportUpdatedClassification.setDisabled(True)

        # Load automatically the CSV file in the pathline in the next tab "Selection of Classification Groups"
        if self.pathLineEdit_meanGroup.currentPath == CSVfilePath:
            self.pathLineEdit_meanGroup.setCurrentPath(" ")
        self.pathLineEdit_meanGroup.setCurrentPath(CSVfilePath)

        return 


    # ---------------------------------------------------- #
    #               Tab: Select Input Data
    #               
    #               Classification Network                 #
    # ---------------------------------------------------- #

    def enableNetwork(self):
        if self.stateCSVDataset and self.stateCSVMeansShape:
            self.pushButton_preprocessData.setEnabled(True)
        else:
            self.pushButton_trainNetwork.setDisabled(True)
            self.pushButton_preprocessData.setDisabled(True)
        return

    def onCSVFileDataset(self):
        # Re-initialization of the dictionary containing the Training dataset
        self.dictShapeModels = dict()

        # Check if the path exists:
        if not os.path.exists(self.pathLineEdit_CSVFileDataset.currentPath):
            self.stateCSVDataset = False
            self.enableNetwork()
            return

        print "------ Selection of a Dataset ------"
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
        self.logic.neuralNetwork.NUM_POINTS = condition4

        self.stateCSVDataset = True
        self.enableNetwork()

        return


    def onCSVFileMeansShape(self):
        # Re-initialization of the dictionary containing the Training dataset
        self.dictGroups = dict()

        # Check if the path exists:
        if not os.path.exists(self.pathLineEdit_CSVFileMeansShape.currentPath):
            self.stateCSVMeansShape = False
            self.enableNetwork()
            return

        print "------ Selection of a Dataset ------"
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
        self.enableNetwork()
        return

    def onPreprocessData(self):
        print "----- onPreprocessData -----"
        self.dictFeatData = dict()
        self.pickle_file = ""

        tempPath = slicer.app.temporaryPath
        # print os.listdir(tempPath)
        
        outputDir = os.path.join(tempPath, "dataFeatures")
        if os.path.isdir(outputDir):
            shutil.rmtree(outputDir)
        os.mkdir(outputDir) 

        #
        # Extract features on shapes, with CondylesFeaturesExtractor
        meansList = ""
        for k, v in self.dictGroups.items():
            if meansList == "":
                meansList = str(v)
            else:
                meansList = meansList + "," +  str(v)

        for group, listvtk in self.dictShapeModels.items():
            for shape in listvtk:
                print shape
# >>>>>>> UNCOMMENT HERE !!! FEATURES EXTRACTION
                self.logic.extractFeatures(shape, meansList, outputDir)

                # # Storage of the means for each group
                self.logic.storageFeaturesData(self.dictFeatData, self.dictShapeModels)


        # 
        # Pickle the data for the network
        self.pickle_file = self.logic.pickleData(self.dictFeatData)
        # print self.pickle_file
        self.pushButton_trainNetwork.setEnabled(True)

        return

    def onTrainNetwork(self):
        print "----- onTrainNetwork -----"
        # self.label_trainNetwork.show()
        self.logic.trainNetworkClassification(self.pickle_file, 'modelCondylesClassification')
        # self.label_trainNetwork.hide()
        
        self.pushButton_exportNetwork.setEnabled(True)
        self.directoryButton_exportNetwork.setEnabled(True)
        return

    def onExportNetwork(self):
        print "----- onExportNetwork -----"

        self.modelName = 'modelCondylesClassification'
        self.logic.exportModelNetwork(self.modelName, self.directoryButton_exportNetwork.directory)
        self.pathLineEdit_networkPath.currentPath = self.directoryButton_exportNetwork.directory + "/coucou.zip"

        return


    def onCSVFileMeansShapeClassify(self):
        # Re-initialization of the dictionary containing the Training dataset
        self.dictShapeModels = dict()

        # Check if the path exists:
        if not os.path.exists(self.pathLineEdit_CSVFileMeansShapeClassify.currentPath):
            return

        # Check if it's a CSV file
        condition1 = self.logic.checkExtension(self.pathLineEdit_CSVFileMeansShapeClassify.currentPath, ".csv")
        if not condition1:
            self.pathLineEdit_CSVFileMeansShapeClassify.setCurrentPath(" ")
            return

        # Read CSV File:
        self.logic.table = self.logic.readCSVFile(self.pathLineEdit_CSVFileMeansShapeClassify.currentPath)
        condition3 = self.logic.creationDictVTKFiles(self.dictShapeModels)
        condition2 = self.logic.checkOneMeshPerGroupInDict(self.dictShapeModels)

        #    If the file is not conformed:
        #    Re-initialization of the dictionary containing the Classification Groups
        if not (condition2 and condition3):
            self.dictShapeModels = dict()
            self.pathLineEdit_CSVFileMeansShapeClassify.setCurrentPath(" ")
            return

        # First condition : All the shapes should have the same number of points
        condition4 = self.logic.checkNumberOfPoints(self.dictShapeModels)
        if not condition4: 
            self.pathLineEdit_CSVFileMeansShapeClassify.setCurrentPath(" ")
            return

        return

    def onNetworkPath(self):
        print "----- onNetworkPath -----"

        # Check qu'il y a un bien:
        #   - modelName.meta
        #   - modelName.index
        #   - modelName.data-00000-of-00001 
        
        # self.modelName = ""

        condition1 = self.logic.checkExtension(self.pathLineEdit_networkPath.currentPath, '.zip')
        if not condition1:
            self.pathLineEdit_networkPath.setCurrentPath(" ")
            return

        # UNZIP FILE DANS LE TEMPPATH

        self.modelName = self.logic.importModelNetwork(self.pathLineEdit_networkPath.currentPath)

        print "c'est dezipper et le model s'appelle :: " + self.modelName

        return


    # Function to select the vtk Input Data
    def onVTKInputData(self):
        # Remove the old vtk file in the temporary directory of slicer if it exists
        if self.patientList:
            print "onVTKInputData remove old vtk file"
            oldVTKPath = slicer.app.temporaryPath + "/" + os.path.basename(self.patientList[0])
            if os.path.exists(oldVTKPath):
                os.remove(oldVTKPath)
        print self.patientList
        # Re-Initialization of the patient list
        self.patientList = list()

        # Handle checkbox "File already in the groups"
        # self.enableOption()

        # Delete the path in CSV file
        currentNode = self.MRMLNodeComboBox_VTKInputData.currentNode()
        if currentNode == None:
            return
        self.pathLineEdit_CSVInputData.setCurrentPath(" ")

        # Adding the vtk file to the list of patient
        currentNode = self.MRMLNodeComboBox_VTKInputData.currentNode()
        if not currentNode == None:
            #     Save the selected node in the temporary directory of slicer
            vtkfilepath = slicer.app.temporaryPath + "/" + self.MRMLNodeComboBox_VTKInputData.currentNode().GetName() + ".vtk"
            self.logic.saveVTKFile(self.MRMLNodeComboBox_VTKInputData.currentNode().GetPolyData(), vtkfilepath)
            #     Adding to the list
            self.patientList.append(vtkfilepath)
        print self.patientList

    # Function to handle the checkbox "File already in the groups"
    # def enableOption(self):
    #     # Enable or disable the checkbox "File already in the groups" according to the data previously selected
    #     currentNode = self.MRMLNodeComboBox_VTKInputData.currentNode()
    #     if currentNode == None:
    #         if self.checkBox_fileInGroups.isChecked():
    #             self.checkBox_fileInGroups.setChecked(False)
    #         self.checkBox_fileInGroups.setDisabled(True)
    #     elif os.path.exists(self.pathLineEdit_NewGroups.currentPath):     # changed to pathLineEdit_previewGroups
    #         self.checkBox_fileInGroups.setEnabled(True)

    #     # Check if the selected file is in the groups used to create the classification groups
    #     self.onCheckFileInGroups()

    # Function to check if the selected file is in the groups used to create the classification groups
    #    - If it's not the case:
    #           - display of a error message
    #           - deselected checkbox
    # def onCheckFileInGroups(self):
    #     if self.checkBox_fileInGroups.isChecked():
    #         node = self.MRMLNodeComboBox_VTKInputData.currentNode()
    #         if not node == None:
    #             vtkfileToFind = node.GetName() + '.vtk'
    #             find = self.logic.actionOnDictionary(self.dictVTKFiles, vtkfileToFind, None, 'find')
    #             if find == False:
    #                 slicer.util.errorDisplay('The selected file is not a file used to create the Classification Groups!')
    #                 self.checkBox_fileInGroups.setChecked(False)

    # Function to select the CSV Input Data
    def onCSVInputData(self):
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
        print self.patientList


        # Handle checkbox "File already in the groups"
        # self.enableOption()

    # Function to define the OA index type of the patient
    #    *** CROSS VALIDATION:
    #    - If the user specified that the vtk file was in the groups used to create the Classification Groups:
    #           - Save the current classification groups
    #           - Re-compute the new classification groups without this file
    #           - Define the OA index type of a patient
    #           - Recovery the classification groups
    #    *** Define the OA index of a patient:
    #    - Else:
    #           - Compute the ShapeOALoads for each group
    #           - Compute the OA index type of a patient
    def onClassifyIndex(self):
        print "------ Compute the OA index Type of a patient ------"

        # Check if the user gave all the data used to compute the OA index type of the patient:
        # - VTK input data or CSV input data
        # - Model network!
        # if not os.path.exists(self.pathLineEdit_selectionClassificationGroups.currentPath):
        #     slicer.util.errorDisplay('Miss the CSV file containing the Classification Groups')
        #     return
        if self.MRMLNodeComboBox_VTKInputData.currentNode() == None and not self.pathLineEdit_CSVInputData.currentPath:
            slicer.util.errorDisplay('Miss the Input Data')
            return

        # **** CROSS VALIDATION ****
        # If the selected file is in the groups used to create the classification groups
        # if self.checkBox_fileInGroups.isChecked():
        #     #      Remove the file in the dictionary used to compute the classification groups
        #     listSaveVTKFiles = list()
        #     vtkfileToRemove = self.MRMLNodeComboBox_VTKInputData.currentNode().GetName() + '.vtk'
        #     listSaveVTKFiles = self.logic.actionOnDictionary(self.dictVTKFiles,
        #                                                      vtkfileToRemove,
        #                                                      listSaveVTKFiles,
        #                                                      'remove')

        #     #      Copy the Classification Groups
        #     dictShapeModelsTemp = dict()
        #     dictShapeModelsTemp = self.dictShapeModels
        #     self.dictShapeModels = dict()

        #     #      Re-compute the new classification groups
        #     self.onComputeNewClassificationGroups()

        # *** Define the OA index type of a patient ***

        self.dictFeatData = dict()
        tempPath = slicer.app.temporaryPath
        networkDir = os.path.join(tempPath, 'Network')
        
        outputDir = os.path.join(tempPath, "dataToClassify")
        if os.path.isdir(outputDir):
            shutil.rmtree(outputDir)
        os.mkdir(outputDir) 

        #
        # Extract features on shapes, with CondylesFeaturesExtractor and get new path (in slicer temp path)
        meansList = ""
        for k, v in self.dictShapeModels.items():
            if meansList == "":
                meansList = str(v)
            else:
                meansList = meansList + "," +  str(v)

        print " :: meansList :: " +  str(meansList)

        for shape in self.patientList:
            # Extract features de la/les shapes a classifier
            # print shape
            self.logic.extractFeatures(shape, meansList, outputDir)

        # Change paths in patientList to have shape with features
        self.logic.storageDataToClassify(self.dictFeatData, self.patientList, outputDir)
        # print self.patientList


        # For each patient:
        self.dictClassified = dict()
        for patient in self.patientList:
            # Compute the classification
            resultgroup = self.logic.evalClassification(self.dictClassified, os.path.join(networkDir, self.modelName), patient)

            # Display the result in the next tab "Result/Analysis"
            self.displayResult(resultgroup, os.path.basename(patient))

        print "\n "
        print self.dictClassified
        # # Remove the CSV file containing the Shape OA Vector Loads
        # self.logic.removeShapeOALoadsCSVFile(self.dictShapeModels.keys())

        # # **** CROSS VALIDATION ****
        # # If the selected file is in the groups used to create the classification groups
        # if self.checkBox_fileInGroups.isChecked():
        #     #      Add the file previously removed to the dictionary used to create the classification groups
        #     self.logic.actionOnDictionary(self.dictVTKFiles,
        #                                   vtkfileToRemove,
        #                                   listSaveVTKFiles,
        #                                   'add')

        #     #      Recovery the Classification Groups previously saved
        #     self.dictShapeModels = dictShapeModelsTemp

        #     #      Remove the data previously created
        #     self.logic.removeDataAfterNCG(self.dictShapeModels)

    # ---------------------------------------------------- #
    #               Tab: Result / Analysis                 #
    # ---------------------------------------------------- #

    # Function to display the result in a table
    def displayResult(self, resultGroup, VTKfilename):
        row = self.tableWidget_result.rowCount
        self.tableWidget_result.setRowCount(row + 1)
        # Column 0: VTK file
        labelVTKFile = qt.QLabel(VTKfilename)
        labelVTKFile.setAlignment(0x84)
        self.tableWidget_result.setCellWidget(row, 0, labelVTKFile)
        # Column 1: Assigned Group
        labelAssignedGroup = qt.QLabel(resultGroup)
        labelAssignedGroup.setAlignment(0x84)
        self.tableWidget_result.setCellWidget(row, 1, labelAssignedGroup)

    # Function to export the result in a CSV File
    def onExportResult(self):
        # Directory
        directory = self.directoryButton_exportResult.directory.encode('utf-8')
        basename = "OAResult.csv"
        # Message if the csv file already exists
        filepath = directory + "/" + basename
        messageBox = ctk.ctkMessageBox()
        messageBox.setWindowTitle(' /!\ WARNING /!\ ')
        messageBox.setIcon(messageBox.Warning)
        if os.path.exists(filepath):
            messageBox.setText('File ' + filepath + ' already exists!')
            messageBox.setInformativeText('Do you want to replace it ?')
            messageBox.setStandardButtons( messageBox.No | messageBox.Yes)
            choice = messageBox.exec_()
            if choice == messageBox.No:
                return

        # Directory
        directory = self.directoryButton_exportResult.directory.encode('utf-8')

        # Store data in a dictionary
        self.logic.creationCSVFileForResult(self.tableWidget_result, directory, basename)

        # Message in the python console and for the user
        print "Export CSV File: " + filepath
        slicer.util.delayDisplay("Result saved")


# ------------------------------------------------------------------------------------ #
#                                   ALGORITHM                                          #
# ------------------------------------------------------------------------------------ #


class ClassificationLogic(ScriptedLoadableModuleLogic):
    def __init__(self, interface):
        self.interface = interface
        self.table = vtk.vtkTable
        self.colorBar = {'Point1': [0, 0, 1, 0], 'Point2': [0.5, 1, 1, 0], 'Point3': [1, 1, 0, 0]}
        self.neuralNetwork = nn.neuralNetwork()

    # Functions to recovery the widget in the .ui file
    def get(self, objectName):
        return self.findWidget(self.interface.widget, objectName)

    def findWidget(self, widget, objectName):
        if widget.objectName == objectName:
            return widget
        else:
            for w in widget.children():
                resulting_widget = self.findWidget(w, objectName)
                if resulting_widget:
                    return resulting_widget
            return None

    # Function to add all the vtk filepaths found in the given directory of a dictionary
    def addGroupToDictionary(self, dictCSVFile, directory, directoryList, group):
        # Fill a dictionary which contains the vtk files for the classification groups sorted by group
        valueList = list()
        for file in os.listdir(directory):
            if file.endswith(".vtk"):
                filepath = directory + '/' + file
                valueList.append(filepath)
        dictCSVFile[group] = valueList

        # Add the path of the directory
        directoryList.insert((group - 1), directory)

    # Function to remove the group of the dictionary
    def removeGroupToDictionary(self, dictCSVFile, directoryList, group):
        # Remove the group from the dictionary
        dictCSVFile.pop(group, None)

        # Remove the path of the directory
        directoryList.pop(group - 1)

    # Check if the path given has the right extension
    def checkExtension(self, filename, extension):
        if os.path.splitext(os.path.basename(filename))[1] == extension:
            return True
        slicer.util.errorDisplay('Wrong extension file, a ' + extension + ' file is needed!')
        return False

    # Function to read a CSV file
    def readCSVFile(self, filename):
        print "CSV FilePath: " + filename
        CSVreader = vtk.vtkDelimitedTextReader()
        CSVreader.SetFieldDelimiterCharacters(",")
        CSVreader.SetFileName(filename)
        CSVreader.SetHaveHeaders(True)
        CSVreader.Update()

        return CSVreader.GetOutput()

    # Function to create a dictionary containing all the vtk filepaths sorted by group
    #    - the paths are given by a CSV file
    #    - If one paths doesn't exist
    #         Return False
    #      Else if all the path of all vtk file exist
    #         Return True
    def creationDictVTKFiles(self, dict):
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

        # Check
        # print "Number of Groups in CSV Files: " + str(len(dict))
        # for key, value in dict.items():
        #     print "Groupe: " + str(key)
        #     print "VTK Files: " + str(value)

        return True

    # Function to check if in each group there is at least more than one mesh
    def checkSeveralMeshInDict(self, dict):
        for key, value in dict.items():
            if type(value) is not ListType or len(value) == 1:
                slicer.util.errorDisplay('The group ' + str(key) + ' must contain more than one mesh.')
                return False
        return True

    # Function to check if in each group there is at least more than one mesh
    def checkOneMeshPerGroupInDict(self, dict):
        for key, value in dict.items():
            if type(value) is ListType: # or len(value) != 1:
                slicer.util.errorDisplay('The group ' + str(key) + ' must contain exactly one mesh.')
                return False
        return True

    # Function to store the shape models for each group in a dictionary
    #    The function return True IF
    #       - all the paths exist
    #       - the extension of the paths is .h5
    #       - there are only one shape model per group
    #    else False
    def creationDictShapeModel(self, dict):
        for i in range(0, self.table.GetNumberOfRows()):
            if not os.path.exists(self.table.GetValue(i,0).ToString()):
                slicer.util.errorDisplay('VTK file not found, path not good at lign ' + str(i+2))
                return False
            if not os.path.splitext(os.path.basename(self.table.GetValue(i,0).ToString()))[1] == '.vtk':
                slicer.util.errorDisplay('Wrong extension file at lign ' + str(i+2) + '. A vtk file is needed!')
                return False
            # if self.table.GetValue(i,1).ToInt() in dict:
            #     slicer.util.errorDisplay('There are more than one shape model (hdf5 file) by groups')
            #     return False
            dict[self.table.GetValue(i,1).ToInt()] = self.table.GetValue(i,0).ToString()

        # Check
        # print "Number of Groups in CSV Files: " + str(len(dict))
        # for key, value in dict.items():
        #     print "Groupe: " + str(key)
        #     print "H5 Files: " + str(value)

        return True

    # Function to add a color map "DisplayClassificationGroup" to all the vtk files
    # which allow the user to visualize each group with a different color in ShapePopulationViewer
    def addColorMap(self, table, dictVTKFiles):
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

    # Function to create a CSV file containing all the selected vtk files that the user wants to display in SPV
    def creationCSVFileForSPV(self, filename, table, dictVTKFiles):
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

    # Function to fill the table of the preview of all VTK files
    #    - Checkable combobox: allow the user to select one or several groups that he wants to display in SPV
    #    - Column 0: filename of the vtk file
    #    - Column 1: combobox with the group corresponding to the vtk file
    #    - Column 2: checkbox to allow the user to choose which models will be displayed in SPV
    #    - Column 3: color that the mesh will have in SPV
    def fillTableForPreviewVTKFilesInSPV(self, dictVTKFiles, checkableComboBox, table):
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
                comboBox.addItems(dictVTKFiles.keys())          # Baisser de 1
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

    # Function to change the group of a vtk file
    #     - The user can change the group thanks to the combobox in the table used for the preview in SPV
    def onComboBoxTableValueChanged(self, dictVTKFiles, table):
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

    # Function to create the same color transfer function than there is in SPV
    def creationColorTransfer(self, groupSelected):
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

    # Function to copy and delete all the arrays of all the meshes contained in a list
    def deleteArrays(self, key, value):
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

    # Function to save a VTK file to the filepath given
    def saveVTKFile(self, polydata, filepath):
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(filepath)
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(polydata)
        else:
            writer.SetInputData(polydata)
        writer.Update()
        writer.Write()

    # Function to save in the temporary directory of Slicer a shape model file called GX.h5
    # built with the vtk files contained in the group X
    def buildShapeModel(self, groupnumber, vtkList):
        print "--- Build the shape model of the group " + str(groupnumber) + " ---"

        # Call of saveModel used to build a shape model from a given list of meshes
        # Arguments:
        #  --groupnumber is the number of the group used to create the shape model
        #  --vtkfilelist is a list of vtk paths of one group that will be used to create the shape model
        #  --resultdir is the path where the newly build model should be saved

        #     Creation of the command line
        scriptedModulesPath = eval('slicer.modules.%s.path' % self.interface.moduleName.lower())
        scriptedModulesPath = os.path.dirname(scriptedModulesPath)
        libPath = os.path.join(scriptedModulesPath)
        sys.path.insert(0, libPath)
        saveModel = os.path.join(scriptedModulesPath, '../hidden-cli-modules/saveModel')
        #saveModel = "/Users/lpascal/Desktop/test/ClassificationExtension-build/bin/saveModel"
        arguments = list()
        arguments.append("--groupnumber")
        arguments.append(groupnumber)
        arguments.append("--vtkfilelist")
        vtkfilelist = ""
        for vtkFiles in vtkList:
            vtkfilelist = vtkfilelist + vtkFiles + ','
        arguments.append(vtkfilelist)
        arguments.append("--resultdir")
        resultdir = slicer.app.temporaryPath
        arguments.append(resultdir)

        #     Call the CLI
        process = qt.QProcess()
        print "Calling " + os.path.basename(saveModel)
        process.start(saveModel, arguments)
        process.waitForStarted()
        # print "state: " + str(process.state())
        process.waitForFinished()
        # print "error: " + str(process.error())

    # Function to compute the mean between all the mesh-files contained in one group
    def computeMean(self, numGroup, vtkList):
        
        print "--- Compute the mean of all the group ---"

        # Call of computeMean used to compute a mean from a shape model
        # Arguments:
        #  --groupnumber is the number of the group used to create the shape model
        #  --resultdir is the path where the newly build model should be saved
        #  --shapemodel: Shape model of one group (H5 file path)

        #     Creation of the command line
        # scriptedModulesPath = eval('slicer.modules.%s.path' % self.interface.moduleName.lower())
        # scriptedModulesPath = os.path.dirname(scriptedModulesPath)
        # libPath = os.path.join(scriptedModulesPath)
        # sys.path.insert(0, libPath)
        # computeMean = os.path.join(scriptedModulesPath, '../hidden-cli-modules/computeMean')
        computeMean = "/Users/prisgdd/Documents/Projects/CNN/computeMean-build/src/bin/computemean"
        arguments = list()
        arguments.append("--inputList")
        vtkfilelist = ""
        for vtkFiles in vtkList:
            vtkfilelist = vtkfilelist + vtkFiles + ','
        arguments.append(vtkfilelist)

        resultdir = slicer.app.temporaryPath
        arguments.append("--outputSurface")
        arguments.append(str(resultdir) + "/meanGroup" + str(numGroup) + ".vtk")

        #     Call the executable
        process = qt.QProcess()
        process.setProcessChannelMode(qt.QProcess.MergedChannels)

        # print "Calling " + os.path.basename(computeMean)
        process.start(computeMean, arguments)
        process.waitForStarted()
        # print "state: " + str(process2.state())
        process.waitForFinished()
        # print "error: " + str(process.error())
        
        processOutput = str(process.readAll())
        # print processOutput


    # Function to remove in the temporary directory all the data used to create the mean for each group
    def removeDataVTKFiles(self, value):
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
        # print dictGroups

    # Function to storage the shape model of each group in a dictionary
    def storeShapeModel(self, dictShapeModels, key):
        filename = "G" + str(key)
        modelPath = slicer.app.temporaryPath + '/' + filename + '.h5'
        dictShapeModels[key] = modelPath

    # Function to create a CSV file:
    #    - Two columns are always created:
    #          - First column: path of the vtk files
    #          - Second column: group associated to this vtk file
    #    - If saveH5 is True, this CSV file will contain a New Classification Group, a thrid column is then added
    #          - Thrid column: path of the shape model of each group
    def creationCSVFile(self, directory, CSVbasename, dictForCSV, option):
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

    # Function to save the data of the new Classification Groups in the directory given by the user
    #       - The mean vtk files of each groups
    #       - The shape models of each groups
    #       - The CSV file containing:
    #               - First column: the paths of mean vtk file of each group
    #               - Second column: the groups associated
    #               - Third column: the paths of the shape model of each group
    def saveNewClassificationGroups(self, basename, directory, dictShapeModels):
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



    # Function to remove in the temporary directory all the data useless after to do a export of the new Classification Groups
    def removeDataAfterNCG(self, dict):
        for key in dict.keys():
            # Remove of the shape model of each group
            # h5Path = slicer.app.temporaryPath + "/G" + str(key) + ".h5"
            path = dict[key]
            if os.path.exists(path):
                # print f
                os.remove(path)

    # Function to make some action on a dictionary
    def actionOnDictionary(self, dict, file, listSaveVTKFiles, action):
        # Action Remove:
        #       Remove the vtk file to the dictionary dict
        #       If the vtk file was found:
        #            Return a list containing the key and the vtk file
        #       Else:
        #            Return False
        # Action Find:
        #       Find the vtk file in the dictionary dict
        #       If the vtk file was found:
        #            Return True
        #       Else:
        #            Return False
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


    def extractFeatures(self, shape, meansList, outputDir):


        # print "--- Extract features of shape : " + shape + " ---"

        # Call of computeMean used to compute a mean from a shape model
        # Arguments:
        #  --groupnumber is the number of the group used to create the shape model
        #  --resultdir is the path where the newly build model should be saved
        #  --shapemodel: Shape model of one group (H5 file path)

        #     Creation of the command line
        # scriptedModulesPath = eval('slicer.modules.%s.path' % self.interface.moduleName.lower())
        # scriptedModulesPath = os.path.dirname(scriptedModulesPath)
        # libPath = os.path.join(scriptedModulesPath)
        # sys.path.insert(0, libPath)
        # computeMean = os.path.join(scriptedModulesPath, '../hidden-cli-modules/computeMean')
        
        condylesfeaturesextractor = "/Users/prisgdd/Documents/Projects/CNN/CondylesFeaturesExtractor-build-cmptemean/src/CondylesFeaturesExtractor/bin/condylesfeaturesextractor"
        
        filename = str(os.path.basename(shape))
        basename, _ = os.path.splitext(filename)
        

        arguments = list()
        arguments.append("--input")
        arguments.append(shape)

        arguments.append("--output")
        arguments.append(str(os.path.join(outputDir,basename)) + "_ft.vtk")

        arguments.append("--meanGroup")
        arguments.append(str(meansList))
        print arguments

        #     Call the executable
        process = qt.QProcess()
        process.setProcessChannelMode(qt.QProcess.MergedChannels)

        # print "Calling " + os.path.basename(computeMean)
        process.start(condylesfeaturesextractor, arguments)
        process.waitForStarted()
        # print "state: " + str(process2.state())
        process.waitForFinished()
        # print "error: " + str(process.error())
        
        processOutput = str(process.readAll())
        print processOutput

        return

    def storageFeaturesData(self, dictFeatData, dictShapeModels):
        for key, value in dictShapeModels.items():
            newValue = list()
            for shape in value:
                filename,_ = os.path.splitext(os.path.basename(shape))
                ftPath = slicer.app.temporaryPath + '/dataFeatures/' + filename + '_ft.vtk'

                newValue.append(ftPath)
            dictFeatData[key] = newValue
        return

    def storageDataToClassify(self, dictFeatData, listPatient, outputDir):
        for i in range(0, len(listPatient)):
            filename,_ = os.path.splitext(os.path.basename(listPatient[i]))
            ftPath = outputDir + "/" + filename + '_ft.vtk'
            listPatient[i] = ftPath
        return listPatient

    def pickleData(self, dictFeatData):
        # for group, vtklist in dictFeatData.items():
        input_Data = inputData.inputData()
        dataset_names = input_Data.maybe_pickle(dictFeatData, 3, force=False)
        self.neuralNetwork.NUM_CLASSES = len(dataset_names)
        self.neuralNetwork.NUM_FEATURES = self.neuralNetwork.NUM_CLASSES + 3 + 4 

        #
        # Determine dataset size
        #
        nbGroups = len(dictFeatData.keys())
        self.neuralNetwork.NUM_CLASSES = nbGroups
        self.neuralNetwork.NUM_FEATURES = nbGroups + 3 + 4 
        small_classe = 100000000
        completeDataset = 0
        for key, value in dictFeatData.items():
            if len(value) < small_classe:
                small_classe = len(value)
            completeDataset = completeDataset + len(value)

        train_size = ( small_classe - 3 ) * nbGroups
        valid_size = 3 * nbGroups
        test_size = completeDataset

        valid_dataset, valid_labels, train_dataset, train_labels = input_Data.merge_datasets(dataset_names, train_size, valid_size) 
        _, _, test_dataset, test_labels = input_Data.merge_all_datasets(dataset_names, test_size)

        train_dataset, train_labels = input_Data.randomize(train_dataset, train_labels)
        valid_dataset, valid_labels = input_Data.randomize(valid_dataset, valid_labels)
        test_dataset, test_labels = input_Data.randomize(test_dataset, test_labels)

        pickle_file = os.path.join(slicer.app.temporaryPath,'condyles.pickle')

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

        #
        #   Fonctions pour le gros du Neural Network
        #   
        #
    def placeholder_inputs(self, batch_size):
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, self.neuralNetwork.NUM_POINTS * self.neuralNetwork.NUM_FEATURES))
        tf_train_labels = tf.placeholder(tf.int32, shape=(batch_size, self.neuralNetwork.NUM_CLASSES))
        return tf_train_dataset, tf_train_labels
        
    ## Reformat into a shape that's more adapted to the models we're going to train:
    #   - data as a flat matrix
    #   - labels as float 1-hot encodings
    def reformat(self, dataset, labels):
        dataset = dataset.reshape((-1, self.neuralNetwork.NUM_POINTS * self.neuralNetwork.NUM_FEATURES)).astype(np.float32)
        labels = (np.arange(self.neuralNetwork.NUM_CLASSES) == labels[:, None]).astype(np.float32)
        return dataset, labels

    ## Reformat into a shape that's more adapted to the models we're going to train:
    #   - data as a flat matrix
    #   - labels as float 1-hot encodings
    def reformat_data(self, dataset):
        dataset = dataset.reshape((-1, self.neuralNetwork.NUM_POINTS * self.neuralNetwork.NUM_FEATURES)).astype(np.float32)
        return dataset
        
    def get_inputs(self, pickle_file):

        # Reoad the data generated in pickleData.py
        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            valid_dataset = save['valid_dataset']
            valid_labels = save['valid_labels']
            # test_dataset = save['test_dataset']
            # test_labels = save['test_labels']
            del save  # hint to help gc free up memory
            print('Training set', train_dataset.shape, train_labels.shape)
            print('Validation set', valid_dataset.shape, valid_labels.shape)
            # print('Test set', test_dataset.shape, test_labels.shape)

            train_dataset, train_labels = self.reformat(train_dataset, train_labels)
            valid_dataset, valid_labels = self.reformat(valid_dataset, valid_labels)
            # test_dataset, test_labels = inputdata.reformat(test_dataset, test_labels)
            print("\nTraining set", train_dataset.shape, train_labels.shape)
            print("Validation set", valid_dataset.shape, valid_labels.shape)
            # print("Test set", test_dataset.shape, test_labels.shape)

            return train_dataset, train_labels, valid_dataset, valid_labels


    def run_training(self, train_dataset, train_labels, valid_dataset, valid_labels, saveModelPath):

        #       >>>>>       A RENDRE GENERIQUE !!!!!!!
        if self.neuralNetwork.NUM_HIDDEN_LAYERS == 1:
            nb_hidden_nodes_1 = 2048
        elif self.neuralNetwork.NUM_HIDDEN_LAYERS == 2:
            nb_hidden_nodes_1, nb_hidden_nodes_2 = 2048, 2048

        # Construct the graph
        graph = tf.Graph()
        with graph.as_default():
            # Input data.
            with tf.name_scope('Inputs_management'):
                # tf_train_dataset, tf_train_labels = placeholder_inputs(self.neuralNetwork.batch_size, name='data')
                tf_train_dataset = tf.placeholder(tf.float32, shape=(self.neuralNetwork.batch_size, self.neuralNetwork.NUM_POINTS * self.neuralNetwork.NUM_FEATURES), name='tf_train_dataset')
                tf_train_labels = tf.placeholder(tf.int32, shape=(self.neuralNetwork.batch_size, self.neuralNetwork.NUM_CLASSES), name='tf_train_labels')

                keep_prob = tf.placeholder(tf.float32, name='keep_prob')

                tf_valid_dataset = tf.constant(valid_dataset, name="tf_valid_dataset")

                # tf_data = tf.Variable(tf.zeros([1,inputdata.NUM_POINTS * inputdata.NUM_FEATURES]))
                tf_data = tf.placeholder(tf.float32, shape=(1,self.neuralNetwork.NUM_POINTS * self.neuralNetwork.NUM_FEATURES), name="input")
                # tf_test_dataset = tf.constant(test_dataset)

            with tf.name_scope('Bias_and_weights_management'):
                weightsDict = self.neuralNetwork.bias_weights_creation(nb_hidden_nodes_1, nb_hidden_nodes_2)    
            
            # Training computation.
            with tf.name_scope('Training_computations'):
                logits, weightsDict = self.neuralNetwork.model(tf_train_dataset, weightsDict)
                
            with tf.name_scope('Loss_computation'):
                loss = self.neuralNetwork.loss(logits, tf_train_labels, self.neuralNetwork.lambda_reg, weightsDict)
            
            
            with tf.name_scope('Optimization'):
                # Optimizer.
                optimizer = tf.train.GradientDescentOptimizer(self.neuralNetwork.learning_rate).minimize(loss)
                # optimizer = tf.train.AdagradOptimizer(self.neuralNetwork.learning_rate).minimize(loss)
            
            # tf.tensor_summary("W_fc1", weightsDict['W_fc1'])
            tf.summary.scalar("Loss", loss)
            summary_op = tf.summary.merge_all()
            saver = tf.train.Saver(weightsDict)

                
            with tf.name_scope('Predictions'):
                # Predictions for the training, validation, and test data.
                train_prediction = tf.nn.softmax(logits)
                valid_prediction = tf.nn.softmax(self.neuralNetwork.model(tf_valid_dataset, weightsDict)[0], name="valid_prediction")

                data_pred = tf.nn.softmax(self.neuralNetwork.model(tf_data, weightsDict)[0], name="output")
                # test_prediction = tf.nn.softmax(self.neuralNetwork.model(tf_test_dataset, weightsDict)[0])


            # -------------------------- #
            #       Let's run it         #
            # -------------------------- #
            # 
            with tf.Session(graph=graph) as session:
                tf.global_variables_initializer().run()
                print("Initialized")

                # create log writer object
                writer = tf.summary.FileWriter('./train', graph=graph)

                for epoch in range(0, self.neuralNetwork.num_epochs):
                    for step in range(self.neuralNetwork.num_steps):
                        # Pick an offset within the training data, which has been randomized.
                        # Note: we could use better randomization across epochs.
                        offset = (step * self.neuralNetwork.batch_size) % (train_labels.shape[0] - self.neuralNetwork.batch_size)
                        # Generate a minibatch.
                        batch_data = train_dataset[offset:(offset + self.neuralNetwork.batch_size), :]
                        batch_labels = train_labels[offset:(offset + self.neuralNetwork.batch_size), :]
                        # Prepare a dictionary telling the session where to feed the minibatch.
                        # The key of the dictionary is the placeholder node of the graph to be fed,
                        # and the value is the numpy array to feed to it.
                        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:0.7}
                        _, l, predictions, summary = session.run([optimizer, loss, train_prediction, summary_op], feed_dict=feed_dict)
                        # _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)


                        # write log
                        batch_count = 20
                        writer.add_summary(summary, epoch * batch_count + step)


                        if (step % 500 == 0):
                            print("Minibatch loss at step %d: %f" % (step, l))
                            print("Minibatch accuracy: %.1f%%" % self.neuralNetwork.accuracy(predictions, batch_labels)[0])
                            print("Validation accuracy: %.1f%%" % self.neuralNetwork.accuracy(valid_prediction.eval(feed_dict = {keep_prob:1.0}), valid_labels)[0])

                # finalaccuracy, mat_confusion, PPV, TPR = self.neuralNetwork.accuracy(test_prediction.eval(feed_dict={keep_prob:1.0}), test_labels)
                # print "\n AVEC DROPOUT\n"
                # print("Test accuracy: %.1f%%" % finalaccuracy)
                # print("\n\nConfusion matrix :\n" + str(mat_confusion))
                # print "\n PPV : " + str(PPV)
                # print "\n TPR : " + str(TPR)

                save_path = saver.save(session, saveModelPath, write_meta_graph=True)
                print("Model saved in file: %s" % save_path)
        
        return 



    def trainNetworkClassification(self, pickle_file, modelName):
        self.neuralNetwork.learning_rate = 0.0005
        self.neuralNetwork.lambda_reg = 0.01
        self.neuralNetwork.num_epochs = 2
        self.neuralNetwork.num_steps =  1001
        self.neuralNetwork.batch_size = 10

        tempPath = slicer.app.temporaryPath
        networkDir = os.path.join(tempPath, "Network")
        if os.path.isdir(networkDir):
            shutil.rmtree(networkDir)
        os.mkdir(networkDir) 

        train_dataset, train_labels, valid_dataset, valid_labels = self.get_inputs(pickle_file)
        saveModelPath = os.path.join(networkDir, modelName)

        self.run_training(train_dataset, train_labels, valid_dataset, valid_labels, saveModelPath)

        return


    def zipdir(self, path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file))

    def exportModelNetwork(self, modelName, directory):
        
        tempPath = slicer.app.temporaryPath
        networkDir = os.path.join(tempPath, 'Network')

        # Zipper tout ca 
        shutil.make_archive(base_name = os.path.join(directory,'coucou'), format = 'zip', root_dir = tempPath, base_dir = 'Network')
        print "jai make_Archiv"

        return

    def importModelNetwork(self, archiveName):

        tempPath = slicer.app.temporaryPath
        networkDir = os.path.join(tempPath, 'Network')
        modelName = ""

        # Si y a deja un Network dans le coin, on le degage
        if os.path.isdir(networkDir):
            shutil.rmtree(networkDir)
        os.mkdir(networkDir) 

        with zipfile.ZipFile(archiveName) as zf:
            zf.extractall(tempPath)
        
        print "jai unpack_Archiv"
        
        modelFound = list()
        listContent = os.listdir(networkDir)

        for file in listContent:
            if os.path.splitext(os.path.basename(file))[1] == ".meta":
                potentialModel = os.path.splitext(os.path.basename(file))[0]
                print potentialModel
                nbPotientalModel = 0

                for fileBis in listContent:
                    if os.path.splitext(os.path.basename(fileBis))[0] == potentialModel:
                        nbPotientalModel = nbPotientalModel + 1
                if nbPotientalModel == 3:
                    modelFound.append(potentialModel)

        
        if len(modelFound) == 1:
            modelName = modelFound[0]
            # C'est parfait, on l'utilise !
            print "Niquel!"
        elif len(modelFound) == 0:
            print " :::: Wallouh y a pas de model dans ton path !!!"
        else:
            print " :: Y a trop de model, il va falloit choisir frere!"

        # La, on update les infos des dimensions! ??

        return modelName


    # # Function in order to compute the shape OA loads of a sample
    # def computeShapeOALoads(self, groupnumber, vtkfilepath, shapemodel):
    #     # Call of computeShapeOALoads used to compute shape loads of a sample for the current shape model
    #     # Arguments:
    #     #  --vtkfile: Sample Input Data (VTK file path)
    #     #  --resultdir: The path where the newly build model should be saved
    #     #  --groupnumber: The number of the group used to create the shape model
    #     #  --shapemodel: Shape model of one group (H5 file path)

    #     #     Creation of the command line
    #     scriptedModulesPath = eval('slicer.modules.%s.path' % self.interface.moduleName.lower())
    #     scriptedModulesPath = os.path.dirname(scriptedModulesPath)
    #     libPath = os.path.join(scriptedModulesPath)
    #     sys.path.insert(0, libPath)
    #     computeShapeOALoads = os.path.join(scriptedModulesPath, '../hidden-cli-modules/computeShapeOALoads')
    #     #computeShapeOALoads = "/Users/lpascal/Desktop/test/ClassificationExtension-build/bin/computeShapeOALoads"
    #     arguments = list()
    #     arguments.append("--groupnumber")
    #     arguments.append(groupnumber)
    #     arguments.append("--vtkfile")
    #     arguments.append(vtkfilepath)
    #     arguments.append("--resultdir")
    #     resultdir = slicer.app.temporaryPath
    #     arguments.append(resultdir)
    #     arguments.append("--shapemodel")
    #     arguments.append(shapemodel)

    #     #     Call the CLI
    #     process = qt.QProcess()
    #     print "Calling " + os.path.basename(computeShapeOALoads)
    #     process.start(computeShapeOALoads, arguments)
    #     process.waitForStarted()
    #     # print "state: " + str(process.state())
    #     process.waitForFinished()
    #     # print "error: " + str(process.error())

    def get_input_shape(self,inputFile):

        # Get features in a matrix (NUM_FEATURES x NUM_POINTS)
        input_Data = inputData.inputData()
        data = input_Data.load_features(inputFile)
        data = data.reshape((-1, self.neuralNetwork.NUM_POINTS * self.neuralNetwork.NUM_FEATURES)).astype(np.float32)
        data = self.reformat_data(data)
        return data

    def get_result(self,prediction):
        return np.argmax(prediction[0,:])

    # Function to compute the OA index of a patient
    def evalClassification(self, dictClassified, model, shape):
        self.neuralNetwork.learning_rate = 0.0005
        self.neuralNetwork.lambda_reg = 0.01
        self.neuralNetwork.num_epochs = 2
        self.neuralNetwork.num_steps =  1001
        self.neuralNetwork.batch_size = 10
        self.neuralNetwork.NUM_POINTS = 1002
        self.neuralNetwork.NUM_CLASSES = 6
        self.neuralNetwork.NUM_FEATURES = self.neuralNetwork.NUM_CLASSES + 3 + 4 
        # Create session, and import existing graph
        # print shape
        myData = self.get_input_shape(shape)
        session = tf.InteractiveSession()


        new_saver = tf.train.import_meta_graph(model + '.meta')
        new_saver.restore(session, model)
        graph = tf.Graph().as_default()
        
        # Get useful tensor in the graph
        tf_data = session.graph.get_tensor_by_name("Inputs_management/input:0")
        data_pred = session.graph.get_tensor_by_name("Predictions/output:0")

        feed_dict = {tf_data: myData}
        data_pred = session.run(data_pred, feed_dict=feed_dict)
        
        result = self.get_result(data_pred)
        print "Shape : " + os.path.basename(shape)
        print "Group predicted :" + str(result) + "\n"
        
        # Mise a jour du dictClassified
        
        # Est-ce que ce result existe deja comme cle ?
        listkey = dictClassified.keys()
        print listkey

        if listkey.count(result):       # La key exist
            valueKey = dictClassified[result]
            valueKey.append(shape)
        else:       # la key n'existe pas 
            valueKey = list()
            valueKey.append(shape)
        dictClassified[result] = valueKey

        return result



        # OAIndexList = list()
        # for key in keyList:
        #     ShapeOAVectorLoadsPath = slicer.app.temporaryPath + "/ShapeOAVectorLoadsG" + str(key) + ".csv"
        #     if not os.path.exists(ShapeOAVectorLoadsPath):
        #         return
        #     tableShapeOAVectorLoads = vtk.vtkTable
        #     tableShapeOAVectorLoads = self.readCSVFile(ShapeOAVectorLoadsPath)
        #     sum = 0
        #     for row in range(0, tableShapeOAVectorLoads.GetNumberOfRows()):
        #         ShapeOALoad = tableShapeOAVectorLoads.GetValue(row, 0).ToDouble()
        #         sum = sum + math.pow(ShapeOALoad, 2)
        #     OAIndexList.append(math.sqrt(sum)/tableShapeOAVectorLoads.GetNumberOfRows())
        # # print OAIndexList
        # resultGroup = OAIndexList.index(min(OAIndexList)) 
        # # print "RESULT: " + str(resultGroup)
        # return resultGroup

    # Function to remove the shape model of each group
    def removeShapeOALoadsCSVFile(self, keylist):
        for key in keylist:
            shapeOALoadsPath = slicer.app.temporaryPath + "/ShapeOAVectorLoadsG" + str(key) + ".csv"
            if os.path.exists(shapeOALoadsPath):
                os.remove(shapeOALoadsPath)

    def creationCSVFileForResult(self, table, directory, CSVbasename):
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

class ClassificationTest(ScriptedLoadableModuleTest):
    pass
