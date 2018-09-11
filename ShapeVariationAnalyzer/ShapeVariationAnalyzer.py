import os, sys
import csv
import unittest
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from types import *
import math
import shutil

#import inputData
import pickle
import numpy as np
import zipfile
import json
import subprocess

#import PythonQt as Qt
from copy import deepcopy

from sklearn.decomposition import PCA
from scipy import stats
from scipy import special


class ShapeVariationAnalyzer(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "ShapeVariationAnalyzer"
        parent.categories = ["Quantification"]
        parent.dependencies = []
        parent.contributors = ["Lopez Mateo (University of North Carolina), Priscille de Dumast (University of Michigan), Laura Pascal (University of Michigan)"]
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

        #self.dictPCA = dict()
        self.PCA_sliders=list()
        self.PCA_sliders_label=list()
        self.PCA_sliders_value_label=list()
        self.PCANode = None

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


        #tab: PCA Analysis
        self.label_valueExploration=self.logic.get('label_valueExploration')
        self.label_varianceExploration=self.logic.get('label_varianceExploration')
        self.label_groupExploration=self.logic.get('label_groupExploration')
        self.label_minVariance=self.logic.get('label_minVariance')
        self.label_maxSlider=self.logic.get('label_maxSlider')
        self.label_colorMode=self.logic.get('label_colorMode')
        self.label_colorModeParam1=self.logic.get('label_colorModeParam1')
        self.label_colorModeParam2=self.logic.get('label_colorModeParam2')

        self.label_normalLabel_1=self.logic.get('label_normalLabel_1')
        self.label_normalLabel_2=self.logic.get('label_normalLabel_2')
        self.label_normalLabel_3=self.logic.get('label_normalLabel_3')
        self.label_normalLabel_4=self.logic.get('label_normalLabel_4')
        self.label_normalLabel_5=self.logic.get('label_normalLabel_5')
        self.label_normalLabel_6=self.logic.get('label_normalLabel_6')
        self.label_normalLabel_7=self.logic.get('label_normalLabel_7')


        self.collapsibleButton_PCA = self.logic.get('collapsibleButton_PCA')

        self.pathLineEdit_CSVFilePCA = self.logic.get('pathLineEdit_CSVFilePCA')  
        self.pathLineEdit_exploration = self.logic.get('pathLineEdit_exploration')

        self.comboBox_groupPCA = self.logic.get('comboBox_groupPCA')
        self.comboBox_colorMode = self.logic.get('comboBox_colorMode')

        self.pushButton_PCA = self.logic.get('pushButton_PCA') 
        self.pushButton_resetSliders = self.logic.get('pushButton_resetSliders')  
        self.pushButton_saveExploration=self.logic.get('pushButton_saveExploration')
        self.pushButton_toggleMean=self.logic.get('pushButton_toggleMean')

        self.label_statePCA = self.logic.get('label_statePCA')

        self.gridLayout_PCAsliders=self.logic.get('gridLayout_PCAsliders')

        self.spinBox_minVariance=self.logic.get('spinBox_minVariance')
        self.spinBox_maxSlider=self.logic.get('spinBox_maxSlider')
        self.spinBox_colorModeParam1=self.logic.get('spinBox_colorModeParam_1')
        self.spinBox_colorModeParam2=self.logic.get('spinBox_colorModeParam_2')

        self.ctkColorPickerButton_groupColor=self.logic.get('ctkColorPickerButton_groupColor')


        #self.doubleSpinBox_insideLimit=self.logic.get('doubleSpinBox_insideLimit')
        #self.doubleSpinBox_insideLimit=self.logic.get('doubleSpinBox_outsidesideLimit')


        # Widget Configuration
        ##PCA exploration Widgets Configuration
        #self.pushButton_PCA.setDisabled(True) 
        #self.comboBox_groupPCA.setDisabled(True)
        self.comboBox_colorMode.addItem('Group color')
        self.comboBox_colorMode.addItem('Unsigned distance to mean shape')
        self.comboBox_colorMode.addItem('Signed distance to mean shape')

        self.spinBox_minVariance.setValue(2)
        self.spinBox_maxSlider.setMinimum(1)
        self.spinBox_maxSlider.setMaximum(8)
        self.spinBox_maxSlider.setValue(8)

        self.spinBox_colorModeParam1.setMinimum(1)
        self.spinBox_colorModeParam2.setMinimum(1)

        self.spinBox_colorModeParam1.setMaximum(10000)
        self.spinBox_colorModeParam2.setMaximum(10000)

        self.spinBox_colorModeParam1.setValue(1)
        self.spinBox_colorModeParam2.setValue(1)
        

        self.label_statePCA.hide()
        self.ctkColorPickerButton_groupColor.color=qt.QColor(255,255,255)
        self.ctkColorPickerButton_groupColor.setDisplayColorName(False)

        self.label_normalLabel_1.hide()
        self.label_normalLabel_2.hide()
        self.label_normalLabel_3.hide()
        self.label_normalLabel_4.hide()
        self.label_normalLabel_5.hide()
        self.label_normalLabel_6.hide()
        self.label_normalLabel_7.hide()

        self.comboBox_groupPCA.hide()
        self.comboBox_colorMode.hide()
        self.ctkColorPickerButton_groupColor.hide()
        self.pushButton_resetSliders.hide()
        self.label_valueExploration.hide()
        self.label_groupExploration.hide()
        self.label_varianceExploration.hide()
        self.pushButton_saveExploration.hide()
        self.pushButton_toggleMean.hide()
        self.spinBox_minVariance.hide()
        self.spinBox_maxSlider.hide()
        self.label_minVariance.hide()
        self.label_maxSlider.hide()

        self.spinBox_colorModeParam1.hide()
        self.spinBox_colorModeParam2.hide()
        self.label_colorMode.hide()
        self.label_colorModeParam1.hide()
        self.label_colorModeParam2.hide()

        #     disable/enable and hide/show widget


        #self.comboBox_healthyGroup.setDisabled(True)
        self.pushButton_exportUpdatedClassification.setDisabled(True)
        self.checkableComboBox_ChoiceOfGroup.setDisabled(True)
        self.tableWidget_VTKFiles.setDisabled(True)
        self.pushButton_previewVTKFiles.setDisabled(True)

        self.label_statePCA.hide()

        self.collapsibleButton_creationCSVFile.setChecked(False)
        self.collapsibleButton_previewClassificationGroups.setChecked(False)


        #     initialisation of the stackedWidget to display the button "add group"
        self.stackedWidget_manageGroup.setCurrentIndex(0)

        #     spinbox configuration in the tab "Creation of CSV File for Classification Groups"
        self.spinBox_group.setMinimum(0)
        self.spinBox_group.setMaximum(0)
        self.spinBox_group.setValue(0)


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
        self.collapsibleButton_PCA.connect('clicked()',
                                            lambda: self.onSelectedCollapsibleButtonOpen(self.collapsibleButton_PCA))


        slicer.mrmlScene.AddObserver(slicer.mrmlScene.EndCloseEvent, self.onCloseScene)
        self.stateCSVMeansShape = False
        self.stateCSVDataset = False


        #       Tab : PCA

        self.pathLineEdit_CSVFilePCA.connect('currentPathChanged(const QString)', self.onCSV_PCA)
        self.pathLineEdit_exploration.connect('currentPathChanged(const QString)', self.onLoadExploration)

        self.pushButton_PCA.connect('clicked()', self.onExportForExploration)
        self.pushButton_resetSliders.connect('clicked()', self.onResetSliders)
        self.pushButton_saveExploration.connect('clicked()',self.onSaveExploration)
        self.pushButton_toggleMean.connect('clicked()',self.onToggleMeanShape)

        self.comboBox_groupPCA.connect('activated(QString)',self.explorePCA)
        self.comboBox_colorMode.connect('activated(QString)',self.onColorModeChange)


        self.spinBox_maxSlider.connect('valueChanged(int)',self.onUpdateSliderList)
        self.spinBox_minVariance.connect('valueChanged(int)',self.onUpdateSliderList)

        self.spinBox_colorModeParam1.connect('valueChanged(int)',self.onUpdateColorModeParam)
        self.spinBox_colorModeParam2.connect('valueChanged(int)',self.onUpdateColorModeParam)

        self.ctkColorPickerButton_groupColor.connect('colorChanged(QColor)',self.onGroupColorChanged)


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


        print("onCloseScene")
        sys.stdout.flush()
        self.dictVTKFiles = dict()
        self.dictGroups = dict()
        self.dictCSVFile = dict()
        self.directoryList = list()
        self.groupSelected = set()
        self.dictShapeModels = dict()
        self.patientList = list()
        self.dictResults = dict()
        self.dictFeatData = dict()

        
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
        self.pushButton_exportUpdatedClassification.setDisabled(True)

        #PCA

        self.label_normalLabel_1.hide()
        self.label_normalLabel_2.hide()
        self.label_normalLabel_3.hide()
        self.label_normalLabel_4.hide()
        self.label_normalLabel_5.hide()
        self.label_normalLabel_6.hide()
        self.label_normalLabel_7.hide()

        self.deletePCASliders()
        self.comboBox_groupPCA.hide()
        self.comboBox_colorMode.hide()
        self.ctkColorPickerButton_groupColor.hide()
        self.pushButton_resetSliders.hide()
        self.label_valueExploration.hide()
        self.label_groupExploration.hide()
        self.label_varianceExploration.hide()
        self.pushButton_saveExploration.hide()
        self.pushButton_toggleMean.hide()
        self.spinBox_minVariance.hide()
        self.spinBox_maxSlider.hide()
        self.label_minVariance.hide()
        self.label_maxSlider.hide()
        self.spinBox_colorModeParam1.hide()
        self.spinBox_colorModeParam2.hide()
        self.label_colorMode.hide()
        self.label_colorModeParam1.hide()
        self.label_colorModeParam2.hide()


        self.pushButton_PCA.setEnabled(False) 
        self.pathLineEdit_CSVFilePCA.setCurrentPath(" ")
        self.pathLineEdit_exploration.setCurrentPath(" ")

        self.spinBox_minVariance.setValue(2)


        # Enable/disable
        self.pushButton_exportUpdatedClassification.setDisabled(True)
        self.checkableComboBox_ChoiceOfGroup.setDisabled(True)
        self.tableWidget_VTKFiles.setDisabled(True)
        self.pushButton_previewVTKFiles.setDisabled(True)

        self.label_statePCA.hide()
        self.stateCSVMeansShape = False
        self.stateCSVDataset = False

        self.collapsibleButton_PCA.setChecked(True)
        self.collapsibleButton_creationCSVFile.setChecked(False)
        self.collapsibleButton_previewClassificationGroups.setChecked(False)

        #     initialisation of the stackedWidget to display the button "add group"
        self.stackedWidget_manageGroup.setCurrentIndex(0)

        #     spinbox configuration in the tab "Creation of CSV File for Classification Groups"
        self.spinBox_group.setMinimum(0)
        self.spinBox_group.setMaximum(0)
        self.spinBox_group.setValue(0)

    def onSelectedCollapsibleButtonOpen(self, selectedCollapsibleButton):
        """  Only one tab can be display at the same time:
        When one tab is opened all the other tabs are closed 
        """
        if selectedCollapsibleButton.isChecked():
            collapsibleButtonList = [self.collapsibleButton_creationCSVFile,
                                     self.collapsibleButton_previewClassificationGroups,
                                     self.collapsibleButton_PCA]
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
        sys.stdout.flush()

        # Load automatically the CSV file in the pathline in the next tab "Creation of New Classification Groups"
        self.pathLineEdit_previewGroups.setCurrentPath(filepath)
        self.pathLineEdit_selectionClassificationGroups.setCurrentPath(filepath)
        #self.pathLineEdit_CSVFileDataset.setCurrentPath(filepath)

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
            launcherSPV = slicer.modules.shapepopulationviewer
            slicer.cli.run(launcherSPV, None, parameters, wait_for_completion=True)

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
        #self.pathLineEdit_CSVFileDataset.setCurrentPath(filepath)

    # ---------------------------------------------------- #
    #               Tab: PCA Analysis                      #
    # ---------------------------------------------------- #

    def onCSV_PCA(self):
        # print("------ onMeanGroupCSV ------")
        self.logic.dictVTKFiles = dict()

        # Check if it's a CSV file
        condition1 = self.logic.checkExtension(self.pathLineEdit_CSVFilePCA.currentPath, ".csv")
        if not condition1:
            self.pathLineEdit_CSVFilePCA.setCurrentPath(" ")
            return

        # Download the CSV file
        self.logic.original_files=self.pathLineEdit_CSVFilePCA.currentPath
        self.logic.table = self.logic.readCSVFile(self.pathLineEdit_CSVFilePCA.currentPath)
        condition2 = self.logic.creationDictVTKFiles(self.logic.dictVTKFiles)
        #condition3 = self.logic.checkOneMeshPerGroupInDict(self.dictVTKFiles)

        # If the file is not conformed:
        #    Re-initialization of the dictionary containing all the data
        #    which will be used to create a new Classification Groups
        if not (condition2):
            self.logic.dictVTKFiles = dict()
            return


        self.pushButton_PCA.setEnabled(True) 

    def onExportForExploration(self):
        """ Function to export the CSV file in the directory chosen by the user
            - Save the CSV file from the dictionary previously filled
            - Load automatically this CSV file in the next tab: "Creation of New Classification Groups"
        """

        self.logic.processPCAForAll(0)

        self.comboBox_groupPCA.setEnabled(True)
        self.comboBox_groupPCA.clear()
        for key, value in self.logic.dictPCA.items():
            group_name = value["group_name"]
            if key != "All":
                self.comboBox_groupPCA.addItem(str(key)+': '+group_name)
            else: 
                self.comboBox_groupPCA.addItem(key)

        self.logic.setCurrentPCAModel(0)
        self.setColorModeSpinBox()
        self.logic.setColorMode(0)

        self.showmean=False
        self.generate3DVisualisationNodes()
        self.generate2DVisualisationNodes()

        index = self.comboBox_colorMode.findText('Group color', qt.Qt.MatchFixedString)
        if index >= 0:
             self.comboBox_colorMode.setCurrentIndex(index)

        self.pathLineEdit_exploration.disconnect('currentPathChanged(const QString)', self.onLoadExploration)
        self.pathLineEdit_exploration.setCurrentPath(' ')
        self.pathLineEdit_exploration.connect('currentPathChanged(const QString)', self.onLoadExploration)

        

        self.explorePCA()



    def onResetSliders(self):
        self.logic.resetPCAPolyData()
        #self.polyDataPCA.Modified()
        for slider in self.PCA_sliders:
            slider.setSliderPosition(0)

    def onChangePCAPolyData(self, num_slider):
        ratio = self.PCA_sliders[num_slider].value

        X=1-(((ratio/1000.0)+1)/2.0)
        self.PCA_sliders_value_label[num_slider].setText(str(round(stats.norm.isf(X),3)))

        self.logic.updatePolyDataExploration(num_slider,ratio)
        #self.polyDataPCA.Modified()

    def onLoadExploration(self):

        JSONfile=self.pathLineEdit_exploration.currentPath
        # Check if the path exists:
        if not os.path.exists(JSONfile):
            return

        # print("------ Creation of a new Classification Groups ------")
        # Check if it's a CSV file
        condition1 = self.logic.checkExtension(JSONfile, ".json")
        if not condition1:
            self.pathLineEdit_previewGroups.setCurrentPath(" ")
            return

        with open(JSONfile,'r') as jsonfile:
            json_dict = json.load(jsonfile)

        PYCfile=json_dict["python_objects_path"]

        with open(PYCfile, 'rb') as pycfile:
            pickle_dict = pickle.load(pycfile)



        self.logic.loadExploration(json_dict,pickle_dict)    


        self.comboBox_groupPCA.setEnabled(True)
        self.comboBox_groupPCA.clear()
        for key, value in self.logic.dictPCA.items():

            group_name = value["group_name"]
            if key != "All":
                self.comboBox_groupPCA.addItem(str(key)+': '+group_name)
            else: 
                self.comboBox_groupPCA.addItem(key)  

        self.logic.setCurrentPCAModel(0)  
        self.setColorModeSpinBox()
        self.logic.setColorMode(0)      
        self.showmean=False

        self.generate3DVisualisationNodes()
        self.generate2DVisualisationNodes()

        index = self.comboBox_colorMode.findText('Group color', qt.Qt.MatchFixedString)
        if index >= 0:
             self.comboBox_colorMode.setCurrentIndex(index)
        #slicer.mrmlScene.RemoveAllDefaultNodes()
        self.explorePCA()

    def onGroupColorChanged(self,newcolor):
        self.logic.changeCurrentGroupColor(newcolor)
        r,g,b=self.logic.getColor()
        displayNode = slicer.mrmlScene.GetFirstNodeByName("PCA Display")
        displayNode.SetColor(r,g,b)
        displayNode.Modified()
        slicer.mrmlScene.GetFirstNodeByName("PCA Exploration").Modified()
        #self.polyDataPCA.Modified()

    def onSaveExploration(self):
        dlg = ctk.ctkFileDialog()
        JSONpath = dlg.getSaveFileName(None, "Export CSV file for Classification groups", os.path.join(qt.QDir.homePath(), "Desktop"), "JSON File (*.json)")

        if JSONpath == '' or JSONpath==' ':
            return
        directory = os.path.dirname(JSONpath)
        basename = os.path.basename(JSONpath)
        name,ext=os.path.splitext(basename)

        PYCpath=os.path.join(directory,name+".pyc")

        min_explained=self.spinBox_minVariance.value/100.0

        json_dict,pickle_dict,polydata_dict=self.logic.extractData()

        for ID,polydata in polydata_dict.items():
            vtkfilepath=os.path.join(directory,'mean'+str(ID)+'.vtk')
            self.logic.saveVTKFile(polydata,vtkfilepath)
            json_dict[ID]["mean_file_path"]=vtkfilepath

        json_dict["original_files"] = self.logic.original_files
        json_dict["python_objects_path"] = PYCpath

        with open(JSONpath,'w') as jsonfile:
            json.dump(json_dict,jsonfile,indent=4)
        print("Export JSON File: " + JSONpath)
        sys.stdout.flush()

        with open(PYCpath,'w') as pycfile:
            pickle.dump(pickle_dict,pycfile)
        print("Export PYC File: " + PYCpath)
        sys.stdout.flush()

        self.pathLineEdit_exploration.disconnect('currentPathChanged(const QString)', self.onLoadExploration)
        self.pathLineEdit_exploration.setCurrentPath(JSONpath)
        self.pathLineEdit_exploration.connect('currentPathChanged(const QString)', self.onLoadExploration)
    
        slicer.util.delayDisplay("Exploration saved")

    def onToggleMeanShape(self):

        if self.showmean==False:
            self.showmean=True
            self.setMeanShapeVisibility()
        else :
            self.showmean=False
            self.setMeanShapeVisibility()

    def onUpdateSliderList(self):
        self.spinBox_maxSlider.value
        self.PCA_sliders
        self.PCA_sliders_label
        self.PCA_sliders_value_label

        ##extract the new number of sliders
        min_explained=self.spinBox_minVariance.value/100.0
        num_components=self.logic.getRelativeNumComponent(min_explained)
        if num_components>self.spinBox_maxSlider.value:
            num_components=self.spinBox_maxSlider.value



        if num_components < len(self.PCA_sliders):
            #print(self.PCA_sliders)
            component_to_delete=len(self.PCA_sliders)-num_components
            for i in range(component_to_delete):
                self.PCA_sliders[i+num_components].deleteLater()
                self.PCA_sliders_label[i+num_components].deleteLater()
                self.PCA_sliders_value_label[i+num_components].deleteLater()
            del self.PCA_sliders[num_components : len(self.PCA_sliders)]
            del self.PCA_sliders_label[num_components : len(self.PCA_sliders_label)]
            del self.PCA_sliders_value_label[num_components : len(self.PCA_sliders_value_label)]
            self.updateVariancePlot(num_components)
            #print(self.PCA_sliders)
        if num_components > len(self.PCA_sliders):
            old_num_components=len(self.PCA_sliders)
            component_to_add=num_components-len(self.PCA_sliders)
            for i in range(component_to_add):
                self.createAndAddSlider(old_num_components+i)
            self.updateVariancePlot(num_components)
  
    def onColorModeChange(self):

        if self.comboBox_colorMode.currentText == 'Group color':


            self.logic.setColorMode(0)
            self.spinBox_colorModeParam1.hide()
            self.spinBox_colorModeParam2.hide()
            self.label_colorModeParam1.hide()
            self.label_colorModeParam2.hide()

        elif self.comboBox_colorMode.currentText == 'Unsigned distance to mean shape':
            self.logic.setColorModeParam(self.spinBox_colorModeParam1.value,self.spinBox_colorModeParam2.value)

            self.label_colorModeParam1.setText('Maximum Distance')
            
            self.spinBox_colorModeParam1.show()
            self.spinBox_colorModeParam2.hide()
            self.label_colorModeParam1.show()
            self.label_colorModeParam2.hide()

            self.logic.setColorMode(1)

        elif self.comboBox_colorMode.currentText == 'Signed distance to mean shape':
            self.logic.setColorModeParam(self.spinBox_colorModeParam1.value,self.spinBox_colorModeParam2.value)

            self.label_colorModeParam1.setText('Maximum Distance Outside')
            self.label_colorModeParam2.setText('Maximum Distance Inside')
            
            self.spinBox_colorModeParam1.show()
            self.spinBox_colorModeParam2.show()
            self.label_colorModeParam1.show()
            self.label_colorModeParam2.show()

            self.logic.setColorMode(2)

        else:
            print('Unexpected color mode option')
        return

    def onUpdateColorModeParam(self):

        self.logic.setColorModeParam(self.spinBox_colorModeParam1.value,self.spinBox_colorModeParam2.value) 
        self.logic.generateDistanceColor()

    def onDataSelected(self, mrlmlPlotSeriesIds, selectionCol):
        for i in range(mrlmlPlotSeriesIds.GetNumberOfValues()):
            Id=mrlmlPlotSeriesIds.GetValue(i)
            plotserienode = slicer.mrmlScene.GetNodeByID(Id)
            if plotserienode.GetName() == "PCA projection":
                #print('Selection detected:')
                valueIds=selectionCol.GetItemAsObject(i)
                Id=valueIds.GetValue(0)

                #table=plotserienode.GetTableNode().GetTable()

                #pc1=table.GetValue(Id,0).ToDouble()
                #pc2=table.GetValue(Id,1).ToDouble()

                self.logic.setCurrentLoadFromPopulation(Id)
                self.explorePCA()      

    def explorePCA(self):

        #Detection of the selected group Id 
        if self.comboBox_groupPCA.currentText == "All":
            keygroup = "All"
        else:
            keygroup = int(self.comboBox_groupPCA.currentText[0])


        #Setting PCA model to use
        self.logic.setCurrentPCAModel(keygroup)

        #get color of the group and set the color picker with this color
        r,g,b=self.logic.getColor()
        self.ctkColorPickerButton_groupColor.color=qt.QColor(int(r*255),int(g*255),int(b*255))

        #setting the maximum number of sliders
        num_components=self.logic.getNumComponent()

        if self.spinBox_maxSlider.value> num_components:
            self.spinBox_maxSlider.setMaximum(num_components)
            self.spinBox_maxSlider.setValue(num_components)
        else:
            self.spinBox_maxSlider.setMaximum(num_components)

        #delete all the previous sliders
        self.deletePCASliders()

        #computing the number of sliders to show
        min_explained=self.spinBox_minVariance.value/100.0
        sliders_number=self.logic.getRelativeNumComponent(min_explained)
        if sliders_number>self.spinBox_maxSlider.value:
            sliders_number=self.spinBox_maxSlider.value

        #create sliders
        for i in range(sliders_number):
            self.createAndAddSlider(i)
            
        #Initialize polydatas for exploration shape and mean shape
        self.logic.initPolyDataMean()
        self.logic.initPolyDataExploration()


        #Initialize the plot view
        self.updateVariancePlot(sliders_number)
        self.updateProjectionPlot()


        #showing QtWidgets

        self.label_normalLabel_1.show()
        self.label_normalLabel_2.show()
        self.label_normalLabel_3.show()
        self.label_normalLabel_4.show()
        self.label_normalLabel_5.show()
        self.label_normalLabel_6.show()
        self.label_normalLabel_7.show()

        self.comboBox_groupPCA.show()
        self.comboBox_colorMode.show()
        self.ctkColorPickerButton_groupColor.show()
        self.pushButton_resetSliders.show()
        self.label_valueExploration.show()
        self.label_groupExploration.show()
        self.label_varianceExploration.show()
        self.pushButton_saveExploration.show()
        self.pushButton_toggleMean.show()
        self.spinBox_minVariance.show()
        self.spinBox_maxSlider.show()
        self.label_minVariance.show()
        self.label_maxSlider.show()

        self.label_colorMode.show()



    def setColorModeSpinBox(self):
        data_std=self.logic.getDataStd()
        std=np.max(data_std)

        self.spinBox_colorModeParam1.disconnect('valueChanged(int)',self.onUpdateColorModeParam)
        self.spinBox_colorModeParam2.disconnect('valueChanged(int)',self.onUpdateColorModeParam)

        self.spinBox_colorModeParam1.setValue(4*std)
        self.spinBox_colorModeParam2.setValue(4*std)

        self.spinBox_colorModeParam1.connect('valueChanged(int)',self.onUpdateColorModeParam)
        self.spinBox_colorModeParam2.connect('valueChanged(int)',self.onUpdateColorModeParam)


    def deletePCASliders(self):
        ##delete all object in the grid
        for i in range(len(self.PCA_sliders)):
            self.PCA_sliders[i].deleteLater()
            self.PCA_sliders_label[i].deleteLater()
            self.PCA_sliders_value_label[i].deleteLater()
        self.PCA_sliders=list()
        self.PCA_sliders_label=list()
        self.PCA_sliders_value_label=list()

    def createAndAddSlider(self,num_slider):
        exp_ratio=self.logic.getExplainedRatio()
        #create the slider
        slider =qt.QSlider(qt.Qt.Horizontal)
        slider.setMaximum(999)
        slider.setMinimum(-999)
        slider.setTickInterval(1)
        position=self.logic.getCurrentRatio(num_slider)
        #print(position)
        slider.setSliderPosition(position)
        #slider.setLayout(self.gridLayout_PCAsliders)
       
        #create the variance ratio label
        label = qt.QLabel()
        label.setText(str(num_slider+1)+':   '+str(round(exp_ratio[num_slider],5)*100)+'%')
        label.setAlignment(qt.Qt.AlignCenter)

        #create the value label
        X=1-(((position/1000.0)+1)/2.0)
        '''if num_slider==4:
            print(X)'''
        valueLabel = qt.QLabel()
        valueLabel.setText(str(round(stats.norm.isf(X),3)))

        #slider and label added to lists
        self.PCA_sliders.append(slider)
        self.PCA_sliders_label.append(label)
        self.PCA_sliders_value_label.append(valueLabel)

        #Slider and label added to the gridLayout
        self.gridLayout_PCAsliders.addWidget(self.PCA_sliders_label[num_slider],num_slider+2,0)
        self.gridLayout_PCAsliders.addWidget(self.PCA_sliders[num_slider],num_slider+2,1)
        self.gridLayout_PCAsliders.addWidget(self.PCA_sliders_value_label[num_slider],num_slider+2,2)
        
        #Connect
        self.PCA_sliders[num_slider].valueChanged.connect(lambda state, x=num_slider: self.onChangePCAPolyData(x))






    #Plots
    def generate2DVisualisationNodes(self):
        #clean previously created nodes
        self.delete2DVisualisationNodes()

        #generate PlotChartNodes to visualize the variance plot and the Projection plot
        variancePlotChartNode = self.generateVariancePlot()
        #variancePlotChartNode.connect('dataSelected(vtkStringArray, vtkCollection)',self.onPointSelect)
        projectionPlotChartNode = self.generateProjectionPlot()

        # Switch to a layout that contains a plot view to create a plot widget
        layoutManager = slicer.app.layoutManager()
        layoutWithPlot = slicer.modules.plots.logic().GetLayoutWithPlot(layoutManager.layout)
        layoutManager.setLayout(layoutWithPlot)

        # Select chart in plot view
        plotWidget = layoutManager.plotWidget(0)
        plotViewNode = plotWidget.mrmlPlotViewNode()

        plotViewNode.SetPlotChartNodeID(projectionPlotChartNode.GetID())
        plotViewNode.SetPlotChartNodeID(variancePlotChartNode.GetID())

        plotView = plotWidget.plotView()
        plotView.dataSelected.disconnect()
        plotView.connect("dataSelected(vtkStringArray*, vtkCollection*)", self.onDataSelected)

    




                    

    def delete2DVisualisationNodes(self):
        node = slicer.mrmlScene.GetFirstNodeByName("PCA projection plot chart")
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)

        node = slicer.mrmlScene.GetFirstNodeByName("PCA variance plot chart")
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)

        node = slicer.mrmlScene.GetFirstNodeByName("PCA projection")
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)

        node = slicer.mrmlScene.GetFirstNodeByName("Variance (%)")
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)

        node = slicer.mrmlScene.GetFirstNodeByName("Sum variance (%)")
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)

        node = slicer.mrmlScene.GetFirstNodeByName("Level 1%")
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)

        node = slicer.mrmlScene.GetFirstNodeByName("Level 95%")
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)

        node = slicer.mrmlScene.GetFirstNodeByName("PCA projection table")
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)

        node = slicer.mrmlScene.GetFirstNodeByName("PCA variance table")
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)

    def generateProjectionPlot(self):
        projectionTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode","PCA projection table")
        table = projectionTableNode.GetTable()

        pc1=vtk.vtkFloatArray()
        pc2=vtk.vtkFloatArray()

        pc1.SetName("pc1")
        pc2.SetName("pc2")

        table.AddColumn(pc1)
        table.AddColumn(pc2)

        #Projection plot serie
        projectionPlotSeries = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "PCA projection")
        projectionPlotSeries.SetAndObserveTableNodeID(projectionTableNode.GetID())
        projectionPlotSeries.SetXColumnName("pc1")
        projectionPlotSeries.SetYColumnName("pc2")
        projectionPlotSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        projectionPlotSeries.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleNone)
        #projectionPlotSeries.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleSquare)
        projectionPlotSeries.SetUniqueColor()

        # Create projection plot chart node
        projectionPlotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode","PCA projection plot chart")
        projectionPlotChartNode.AddAndObservePlotSeriesNodeID(projectionPlotSeries.GetID())
        projectionPlotChartNode.SetTitle('Population projection')
        projectionPlotChartNode.SetXAxisTitle('pc1')
        projectionPlotChartNode.SetYAxisTitle('pc2')

        return projectionPlotChartNode

    def generateVariancePlot(self):
    
        varianceTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode","PCA variance table")
        table = varianceTableNode.GetTable()

        x=vtk.vtkFloatArray()
        evr=vtk.vtkFloatArray()
        sumevr=vtk.vtkFloatArray()
        level95=vtk.vtkFloatArray()
        level1=vtk.vtkFloatArray()

        x.SetName("Component")
        evr.SetName("ExplainedVarianceRatio")
        sumevr.SetName("SumExplainedVarianceRatio")
        level95.SetName("level95%")
        level1.SetName("level1%")

        table.AddColumn(x)
        table.AddColumn(evr)
        table.AddColumn(sumevr)
        table.AddColumn(level95)
        table.AddColumn(level1)
        #level1
        level1PlotSeries = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "Level 1%")
        level1PlotSeries.SetAndObserveTableNodeID(varianceTableNode.GetID())
        level1PlotSeries.SetXColumnName("Component")
        level1PlotSeries.SetYColumnName("level1%")
        level1PlotSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        level1PlotSeries.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone)
        level1PlotSeries.SetUniqueColor()

        #level95
        level95PlotSeries = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "Level 95%")
        level95PlotSeries.SetAndObserveTableNodeID(varianceTableNode.GetID())
        level95PlotSeries.SetXColumnName("Component")
        level95PlotSeries.SetYColumnName("level95%")
        level95PlotSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        level95PlotSeries.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleNone)
        level95PlotSeries.SetUniqueColor()

        #Sum Explained Variance plot serie
        sumevrPlotSeries = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "Sum variance (%)")
        sumevrPlotSeries.SetAndObserveTableNodeID(varianceTableNode.GetID())
        sumevrPlotSeries.SetXColumnName("Component")
        sumevrPlotSeries.SetYColumnName("SumExplainedVarianceRatio")
        sumevrPlotSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        sumevrPlotSeries.SetUniqueColor()

        #Explained Variance plot serie
        evrPlotSeries = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "Variance (%)")
        evrPlotSeries.SetAndObserveTableNodeID(varianceTableNode.GetID())
        evrPlotSeries.SetXColumnName("Component")
        evrPlotSeries.SetYColumnName("ExplainedVarianceRatio")
        evrPlotSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatterBar)
        evrPlotSeries.SetUniqueColor()

        # Create variance plot chart node
        plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode","PCA variance plot chart")
        plotChartNode.AddAndObservePlotSeriesNodeID(evrPlotSeries.GetID())
        plotChartNode.AddAndObservePlotSeriesNodeID(sumevrPlotSeries.GetID())
        plotChartNode.AddAndObservePlotSeriesNodeID(level95PlotSeries.GetID())
        plotChartNode.AddAndObservePlotSeriesNodeID(level1PlotSeries.GetID())
        plotChartNode.SetTitle('Explained Variance Ratio')
        plotChartNode.SetXAxisTitle('Component')
        plotChartNode.SetYAxisTitle('Explained Variance Ratio') 

        return plotChartNode

    def updateVariancePlot(self,num_components):

        varianceTableNode = slicer.mrmlScene.GetFirstNodeByName("PCA variance table")
        table = varianceTableNode.GetTable()
        table.Initialize()

        level95 , level1=self.logic.getPlotLevel(num_components)
        level95.SetName("level95%")
        level1.SetName("level1%")
        table.AddColumn(level95)
        table.AddColumn(level1)

        x,evr,sumevr= self.logic.getPCAVarianceExplainedRatio(num_components)
        x.SetName("Component")
        evr.SetName("ExplainedVarianceRatio")
        sumevr.SetName("SumExplainedVarianceRatio")

        table.AddColumn(x)
        table.AddColumn(evr)
        table.AddColumn(sumevr)

    def updateProjectionPlot(self):   
        projectionTableNode = slicer.mrmlScene.GetFirstNodeByName("PCA projection table")
        table = projectionTableNode.GetTable()
        table.Initialize()

        pc1,pc2=self.logic.getPCAProjections()

        pc1.SetName("pc1")
        pc2.SetName("pc2")

        table.AddColumn(pc1)
        table.AddColumn(pc2) 


    #polydata

    def generate3DVisualisationNodes(self):
        self.delete3DVisualisationNodes()
        ##For Mean shape
        #clear scene from previous PCA exploration
        
        #create Model Node
        PCANode = slicer.vtkMRMLModelNode()
        PCANode.SetAndObservePolyData(self.logic.polydataMean)
        PCANode.SetName("PCA Mean")
        #create display node
        R,G,B=self.logic.getColor()
        modelDisplay = slicer.vtkMRMLModelDisplayNode()
        modelDisplay.SetColor(0.5,0.5,0.5) 
        modelDisplay.SetOpacity(0.8)
        #modelDisplay.SetBackfaceCulling(0)
        modelDisplay.SetScene(slicer.mrmlScene)
        modelDisplay.SetName("PCA Mean Display")
        modelDisplay.VisibilityOff()
        
        slicer.mrmlScene.AddNode(modelDisplay)
        PCANode.SetAndObserveDisplayNodeID(modelDisplay.GetID())

        slicer.mrmlScene.AddNode(PCANode)

        self.setMeanShapeVisibility()

        ##For Exploration
        #clear scene from previous PCA exploration
        
        #create Model Node
        PCANode = slicer.vtkMRMLModelNode()
        PCANode.SetAndObservePolyData(self.logic.polydataExploration)
        PCANode.SetName("PCA Exploration")
        #create display node
        R,G,B=self.logic.getColor()
        modelDisplay = slicer.vtkMRMLModelDisplayNode()
        modelDisplay.SetColor(R,G,B) 
        modelDisplay.SetOpacity(1)
        modelDisplay.AutoScalarRangeOff()
        #modelDisplay.SetBackfaceCulling(0)
        modelDisplay.SetScene(slicer.mrmlScene)
        modelDisplay.SetName("PCA Display")

        signedcolornode=self.logic.generateSignedDistanceLUT()
        unsignedcolornode=self.logic.generateUnsignedDistanceLUT()

        signedcolornode.SetName('PCA Signed Distance Color Table')
        unsignedcolornode.SetName('PCA Unsigned Distance Color Table')

        slicer.mrmlScene.AddNode(signedcolornode)
        slicer.mrmlScene.AddNode(unsignedcolornode)

        slicer.mrmlScene.AddNode(modelDisplay)
        PCANode.SetAndObserveDisplayNodeID(modelDisplay.GetID())

        slicer.mrmlScene.AddNode(PCANode)

    def delete3DVisualisationNodes(self):
        node = slicer.mrmlScene.GetFirstNodeByName("PCA Exploration")
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)
        node = slicer.mrmlScene.GetFirstNodeByName("PCA Display")
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)
        node = slicer.mrmlScene.GetFirstNodeByName("PCA Mean")
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)
        node = slicer.mrmlScene.GetFirstNodeByName("PCA Mean Display")
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)
        node = slicer.mrmlScene.GetFirstNodeByName('PCA Signed Distance Color Table')
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)
        node = slicer.mrmlScene.GetFirstNodeByName('PCA Unsigned Distance Color Table')
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)

    def setMeanShapeVisibility(self):
        node = slicer.mrmlScene.GetFirstNodeByName("PCA Mean Display")
        if self.showmean==False:
            node.VisibilityOff()
        else :
            node.VisibilityOn()




# ------------------------------------------------------------------------------------ #
#                                   ALGORITHM                                          #
# ------------------------------------------------------------------------------------ #

class ShapeVariationAnalyzerLogic(ScriptedLoadableModuleLogic):
    def __init__(self, interface):
        self.interface = interface
        self.table = vtk.vtkTable
        self.colorBar = {'Point1': [0, 0, 1, 0], 'Point2': [0.5, 1, 1, 0], 'Point3': [1, 1, 0, 0]}
        
        #Exploration variable
  
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
        sys.stdout.flush()
        if caller.IsA('vtkMRMLCommandLineModuleNode'):
            print("Status is %s" % caller.GetStatusString())
            sys.stdout.flush()
            # print("output:   \n %s" % caller.GetOutputText())
            # print("error:   \n %s" % caller.GetErrorText())
        return

    def computeMean(self, numGroup, vtkList):
        """ Function to compute the mean between all 
        the mesh-files contained in one group
        """
        print("--- Compute the mean of all the group ---")
        sys.stdout.flush()
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



    #################
    # PCA ALGORITHM #       
    #################

    #computing PCA
    def readPCAData(self, fileList):
        """
        Read data from fileList and format it for PCA computation
        """

        y_design = []
        numpoints = -1
        nshape = 0
        polydata = 0
        t=1
        group_name=None
        ID=0

        for vtkfile in fileList:
            print(ID,vtkfile)
            ID+=1

            if vtkfile.endswith((".vtk")):
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
                    print("WARNING! File ignored, the number of points is not the same for the shape:", vtkfilename)
                    sys.stdout.flush()
                    pass

                for i in range(shapedatapoints.GetNumberOfPoints()):
                    p = shapedatapoints.GetPoint(i)
                    y_design[nshape].append(p)
                nshape+=1
                
        y_design = np.array(y_design)
        return y_design.reshape(y_design.shape[0], -1),polydata,group_name

    def processPCA(self,X,min_explained ,group_name):
        X_ = np.mean(X, axis=0, keepdims=True)
        X_std = np.std(X,axis=0,keepdims=True)


        pca = PCA()
        pca.fit(X - X_)

        sum_explained = 0.0
        num_components = 0
        
        for evr in pca.explained_variance_ratio_:
            if evr < min_explained:
                break
            num_components += 1

        #print('num_comp = ',num_components)

        pca = PCA(n_components=num_components)
        X_pca=pca.fit_transform(X - X_)

        X_pca_mean = np.mean(X_pca, axis=0, keepdims=True)
        X_pca_var = np.std(X_pca, axis=0, keepdims=True)

        pca_model = {}
        pca_model["pca"] = pca
        pca_model['explained_variance_ratio']=pca.explained_variance_ratio_
        pca_model["eigenvalues"]=np.multiply(pca.singular_values_,pca.singular_values_)
        pca_model["components"]=pca.components_
        pca_model["num_components"]=num_components
        pca_model["data_mean"] = X_
        pca_model["data_std"] = X_std
        pca_model["data_projection"]=X_pca
        pca_model["data_projection_mean"]=X_pca_mean[0]
        pca_model["data_projection_var"]=X_pca_var[0]
        pca_model["current_pca_loads"] = np.zeros(num_components) 
        pca_model["group_name"]=group_name
        pca_model["color"]=(1,1,1)

        return pca_model


    def processPCAForAll(self,min_explained):
        #clean PCAdata dict
        self.dictPCA = dict()
        print("min explained", min_explained)
        sys.stdout.flush()

        all_data=None
        #for each group, compute PCA
        for key, value in self.dictVTKFiles.items():
            #read data of the group
            data ,polydata,group_name = self.readPCAData(value)
            #store data
            if all_data is None:
                all_data=deepcopy(data)
            else:
                all_data=np.concatenate((all_data,data),axis=0)
            #compute PCA
            pca_model=self.processPCA(data,min_explained,group_name)
            #PCA model stored in a dict
            self.dictPCA[key]=pca_model

        #compute PCA for all the data
        pca_model=self.processPCA(all_data,min_explained,"All")
        self.dictPCA["All"]=pca_model

        self.polydata=polydata
        self.polydataMean=vtk.vtkPolyData()
        self.polydataMean.DeepCopy(polydata)
        self.polydataExploration=vtk.vtkPolyData()
        self.polydataExploration.DeepCopy(polydata)
    

    #load, save, format data
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


    #Common
    def readVTKfile(self,filename):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()
        data = reader.GetOutput()
        return data


    #Plots
    def getPCAProjections(self):
        X_pca = self.current_pca_model["data_projection"]

        pc1 = X_pca[:,0].flatten()
        pc1 = vtk.util.numpy_support.numpy_to_vtk(num_array=pc1, array_type=vtk.VTK_FLOAT)  

        pc2 = X_pca[:,1].flatten()
        pc2 = vtk.util.numpy_support.numpy_to_vtk(num_array=pc2, array_type=vtk.VTK_FLOAT)    

        return pc1, pc2

    def getPlotLevel(self,num_component):
        
        level95=np.ones(num_component)*95
        level1=np.ones(num_component)
        #xlevel=vtk.util.numpy_support.numpy_to_vtk(num_array=xlevel, array_type=vtk.VTK_FLOAT)
        level95=vtk.util.numpy_support.numpy_to_vtk(num_array=level95, array_type=vtk.VTK_FLOAT)
        level1=vtk.util.numpy_support.numpy_to_vtk(num_array=level1, array_type=vtk.VTK_FLOAT)

        return  level95, level1

    def getPCAVarianceExplainedRatio(self,num_component):
        evr = self.current_pca_model['explained_variance_ratio'][0:num_component].flatten()*100
        sumevr = np.cumsum(evr)
        evr = vtk.util.numpy_support.numpy_to_vtk(num_array=evr, array_type=vtk.VTK_FLOAT)
        sumevr=vtk.util.numpy_support.numpy_to_vtk(num_array=sumevr, array_type=vtk.VTK_FLOAT)

        x = np.arange(1,num_component+1).flatten()
        x = vtk.util.numpy_support.numpy_to_vtk(num_array=x, array_type=vtk.VTK_FLOAT)

        return x,evr,sumevr

    #polydata

    def setCurrentLoadFromPopulation(self,Id):
        population_projection=self.current_pca_model["data_projection"]
        #if Id == 141:
         #   print(population_projection[Id,:])
        self.current_pca_model["current_pca_loads"] = deepcopy(population_projection[Id,:])

    def initPolyDataExploration(self):
        PCA_model=self.current_pca_model['pca']
        PCA_current_loads = self.current_pca_model["current_pca_loads"]     
        mean =self.current_pca_model['data_mean']

        self.pca_points_numpy=PCA_model.inverse_transform(PCA_current_loads)+mean



        self.pca_exploration_points=self.generateVTKPointsFromNumpy(self.pca_points_numpy[0])
        self.polydataExploration.SetPoints(self.pca_exploration_points)
        
        

        self.autoOrientNormals(self.polydataExploration)
        self.generateDistanceColor()

        self.polydataExploration.Modified()

    def initPolyDataMean(self):
        mean =self.current_pca_model['data_mean']

        mean_points=self.generateVTKPointsFromNumpy(mean[0])

        self.polydataMean.SetPoints(mean_points)

        self.autoOrientNormals(self.polydataMean)

        self.polydataMean.Modified()

    def resetPCAPolyData(self):
        num_components=self.current_pca_model["num_components"]
        self.current_pca_model["current_pca_loads"] = np.zeros(num_components) 
        self.pca_points_numpy=self.current_pca_model['data_mean']

        self.modifyVTKPointsFromNumpy(self.pca_points_numpy[0])
        self.autoOrientNormals(self.polydataExploration)
        self.polydataExploration.Modified()

    def updatePolyDataExploration(self,num_slider,ratio):


        

        #update current loads
        pca_mean=self.current_pca_model["data_projection_mean"]
        pca_var=self.current_pca_model["data_projection_var"]

        PCA_model=self.current_pca_model['pca']
        PCA_current_loads = self.current_pca_model["current_pca_loads"] 

        mean =self.current_pca_model['data_mean']

        X=1-(((ratio/1000.0)+1)/2.0)

        PCA_current_loads[num_slider]=pca_mean[num_slider]+stats.norm.isf(X)*pca_var[num_slider]
        #print(self.PCA_current_loads) 
        sys.stdout.flush()

        self.pca_points_numpy=PCA_model.inverse_transform(PCA_current_loads)+mean


        self.modifyVTKPointsFromNumpy(self.pca_points_numpy[0])

        self.generateDistanceColor()

        self.autoOrientNormals(self.polydataExploration)
        self.polydataExploration.Modified()
        
        #print(pca_points.reshape(1002,3))
        #sys.stdout.flush()
        #return  self.current_pca_model["polydata"]

    def generateMeanShape(self,ID):

        mean =self.dictPCA[ID]['data_mean']


        mean_mesh=self.generateVTKPointsFromNumpy(mean[0])

        polydata=vtk.vtkPolyData()

        polydata.DeepCopy(self.polydata)
        polydata.SetPoints(mean_mesh)

        self.autoOrientNormals(polydata)

        return polydata


    def disableExplorationScalarView(self):
        model1=slicer.mrmlScene.GetFirstNodeByName('PCA Exploration')

        if model1 is not None:
            model1.GetDisplayNode().SetScalarVisibility(0)
            #model1.GetDisplayNode().SetScalarVisibility(1)

            model1.Modified()

    def enableExplorationScalarView(self):
        exploration_node=slicer.mrmlScene.GetFirstNodeByName('PCA Exploration')
        exploration_node.GetDisplayNode().SetActiveScalarName('Distance')
        exploration_node.GetDisplayNode().SetScalarVisibility(1)




        exploration_node.Modified()

    def setColorMode(self,colormode):
        self.colormode = colormode

        if colormode == 1: #unsigned distance
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
                explorationnode.Modified()

        self.generateDistanceColor()

    def generateDistanceColor(self):

        #print(self.colormode)

        if self.colormode==0:
            self.disableExplorationScalarView()
            return

        if self.colormode==1:
            mean =self.current_pca_model['data_mean'][0]
            exploration_points=self.pca_points_numpy[0]

            max_dist=self.colormodeparam1

            node = slicer.mrmlScene.GetFirstNodeByName("PCA Display")
            node.SetScalarRange(0,100*max_dist)

            colors=self.unsignedDistance(mean,exploration_points)

        if self.colormode==2:
            mean =self.current_pca_model['data_mean'][0]
            exploration_points=self.pca_points_numpy[0]

            max_dist_inside=self.colormodeparam2
            max_dist_outside=self.colormodeparam1

            node = slicer.mrmlScene.GetFirstNodeByName("PCA Display")
            node.SetScalarRange(-100*max_dist_inside,100*max_dist_outside)

            colors=self.signedDistance(mean,exploration_points)

        self.polydataExploration.GetPointData().SetScalars(colors)
        self.polydataExploration.GetPointData().Modified()
        self.polydataExploration.Modified()
        self.enableExplorationScalarView()

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
        #select_enclosed_points.CheckSurfaceOn()
        select_enclosed_points.SetInputData(self.polydataExploration)
        select_enclosed_points.SetSurfaceData(self.polydataMean)
        select_enclosed_points.SetTolerance(0.000001)
        select_enclosed_points.Update()
        

        for i in range(0,len(mean),3):
            point=exploration_points[i:i+3]
            meanpoint=mean[i:i+3]
            distance = np.linalg.norm(point-meanpoint)

            if select_enclosed_points.IsInside(i/3) == 1:
                ratio=distance/max_inside
                color=ratio*blue+(1-ratio)*white
            else:
                ratio=distance/max_outside
                color=ratio*red+(1-ratio)*white

            colors.InsertNextTuple(100*color)#ratio*blue+(1-ratio)*white)
            

        colors.Modified()
        return colors

    def unsignedDistance(self,mean,exploration_points):
        colors = vtk.vtkFloatArray()
        colors.SetName("Distance")
        colors.SetNumberOfComponents(1)

        color=np.array([0])

        for i in range(0,len(mean),3):
            point=exploration_points[i:i+3]
            meanpoint=mean[i:i+3]
            distance = np.linalg.norm(point-meanpoint)

            color[0]=100*distance#100*(1-ratio)
            
            colors.InsertNextTuple(color)#ratio*blue+(1-ratio)*white)

        colors.Modified()
        return colors

    def generateUnsignedDistanceLUT(self):
        number_of_color=255
        colorTableNode = slicer.vtkMRMLColorTableNode()
        colorTableNode.SetName('PCA Unsigned Distance Color Table')
        colorTableNode.SetTypeToUser()
        colorTableNode.HideFromEditorsOff()
        colorTableNode.SaveWithSceneOff()
        colorTableNode.SetNumberOfColors(number_of_color)
        colorTableNode.GetLookupTable().SetTableRange(0,number_of_color-1)


        c=1
        for i in range(number_of_color):
            colorTableNode.AddColor(str(i), 1, c, c, 1)    
            c = c- 1.0/number_of_color           
        
        return colorTableNode

    def generateSignedDistanceLUT(self):
        number_of_color=255
        colorTableNode = slicer.vtkMRMLColorTableNode()
        colorTableNode.SetName('PCA Signed Distance Color Table')
        colorTableNode.SetTypeToUser()
        colorTableNode.HideFromEditorsOff()
        colorTableNode.SaveWithSceneOff()
        colorTableNode.SetNumberOfColors(number_of_color)
        colorTableNode.GetLookupTable().SetTableRange(0,number_of_color-1)

        blueshade=0
        redshade=1
        for i in range(number_of_color):
            if blueshade <= 1:
                colorTableNode.AddColor(str(i), blueshade, blueshade,1 , 1)  
                #print(str(i), blueshade, blueshade,1 , 1)  
                blueshade = blueshade+ 2.0/number_of_color 
            else:
                colorTableNode.AddColor(str(i), 1, redshade, redshade, 1)    
                #print(str(i), 1, redshade, redshade, 1)
                redshade = redshade- 2.0/number_of_color

        return colorTableNode


    def modifyVTKPointsFromNumpy(self,npArray):
        num_points = npArray.shape[0]/3
        for i in range(num_points):
            self.pca_exploration_points.SetPoint(i,npArray[3*i],npArray[3*i+1],npArray[3*i+2])
        self.pca_exploration_points.Modified()

    def generateVTKPointsFromNumpy(self,npArray):
        num_points = npArray.shape[0]/3
        vtk_points = vtk.vtkPoints()
        for i in range(num_points):
            vtk_points.InsertNextPoint(npArray[3*i],npArray[3*i+1],npArray[3*i+2])
        return vtk_points

    def autoOrientNormals(self, model):
        #surface = None
        #surface = model.GetPolyDataConnection()
        normals = vtk.vtkPolyDataNormals()
        normals.SetAutoOrientNormals(True)
        normals.SetFlipNormals(False)
        normals.SetSplitting(False)
        normals.ConsistencyOn()
        normals.SetInputData(model)
        normals.Update()
        normalspoint=normals.GetOutput().GetPointData().GetArray("Normals")
        model.GetPointData().SetNormals(normalspoint)

        '''normalspoint=normals.GetOutput().GetPointData().GetArray("Normals")
        normalspoint.SetName("Normals")
        model.GetPointData().RemoveArray("Normals")
        model.GetPointData().AddArray(normalspoint)

        normalscell=normals.GetOutput().GetCellData().GetArray("Normals")
        normalscell.SetName("Normals")
        model.GetCellData().RemoveArray("Normals")
        model.GetCellData().AddArray(normalscell)'''
        return normals.GetOutput()

    #group color
    def getColor(self):
        (r,g,b)=self.current_pca_model['color']
        return r,g,b

    def changeCurrentGroupColor(self,color):
        color=(color.red()/255.0,color.green()/255.0,color.blue()/255.0)
        self.current_pca_model['color']=color


    def setColorModeParam(self,param1,param2):
        self.colormodeparam1=param1
        self.colormodeparam2=param2

    def setCurrentPCAModel(self, keygroup):

        self.current_pca_model=self.dictPCA[keygroup]

    def getCurrentRatio(self,num_slider):
        pca_mean=self.current_pca_model["data_projection_mean"][num_slider]
        pca_var=self.current_pca_model["data_projection_var"][num_slider]

        PCA_current_loads = self.current_pca_model["current_pca_loads"] [num_slider]

        '''X=1-(((ratio/1000.0)+1)/2.0)
        stats.norm.isf(X)
        PCA_current_loads[num_slider]=pca_mean[num_slider]+stats.norm.isf(X)*pca_var[num_slider]'''

        '''if num_slider==4:
            print((PCA_current_loads-pca_mean)/(pca_var))'''

        ratio = (PCA_current_loads-pca_mean)/(pca_var)
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





class ShapeVariationAnalyzerTest(ScriptedLoadableModuleTest):
    pass
