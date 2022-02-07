from __future__ import print_function
from __future__ import division
import os, sys
import csv
import unittest
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from types import *
import math
import shutil


import pickle
import numpy as np
import zipfile
import json
import subprocess


from copy import deepcopy

from scipy import stats

import time

import shapepcalib as shapca
from cpns.cpns import CPNS


class ShapeVariationAnalyzer(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Population Analysis"
        parent.categories = ["Shape Analysis"]
        parent.dependencies = []
        parent.contributors = ["Lopez Mateo (University of North Carolina), Priscille de Dumast (University of Michigan), Laura Pascal (University of Michigan)"]
        parent.helpText = """
            Shape Variation Analyzer allows the PCA decomposition and exploration of 3D models. 
            The generated models can be evaluated by computing their specificity, compactness and generalization.
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
        self.logic = ShapeVariationAnalyzerLogic()

        #print(dir(self.logic))
        
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
        self.collapsibleButton_creationCSVFile = self.getUI('CollapsibleButton_creationCSVFile')
        self.spinBox_group = self.getUI('spinBox_group')
        self.directoryButton_creationCSVFile = self.getUI('DirectoryButton_creationCSVFile')
        self.stackedWidget_manageGroup = self.getUI('stackedWidget_manageGroup')
        self.pushButton_addGroup = self.getUI('pushButton_addGroup')
        self.pushButton_removeGroup = self.getUI('pushButton_removeGroup')
        self.pushButton_modifyGroup = self.getUI('pushButton_modifyGroup')
        self.pushButton_exportCSVfile = self.getUI('pushButton_exportCSVfile')

        #          Tab: Creation of New Classification Groups
        self.collapsibleButton_previewClassificationGroups = self.getUI('CollapsibleButton_previewClassificationGroups')
        self.pathLineEdit_previewGroups = self.getUI('pathLineEdit_previewGroups')
        self.pathLineEdit_previewGroups.filters = ctk.ctkPathLineEdit.Files
        self.pathLineEdit_previewGroups.nameFilters = ['*.csv']
        self.collapsibleGroupBox_previewVTKFiles = self.getUI('CollapsibleGroupBox_previewVTKFiles')
        self.checkableComboBox_ChoiceOfGroup = self.getUI('CheckableComboBox_ChoiceOfGroup')
        self.tableWidget_VTKFiles = self.getUI('tableWidget_VTKFiles')
        self.pushButton_previewVTKFiles = self.getUI('pushButton_previewVTKFiles')
        self.pushButton_exportUpdatedClassification = self.getUI('pushButton_exportUpdatedClassification')

        #          Tab: PCA Analysis
        self.label_valueExploration=self.getUI('label_valueExploration')
        self.label_varianceExploration=self.getUI('label_varianceExploration')
        self.label_groupExploration=self.getUI('label_groupExploration')
        self.label_minVariance=self.getUI('label_minVariance')
        self.label_maxSlider=self.getUI('label_maxSlider')
        self.label_colorMode=self.getUI('label_colorMode')
        self.label_colorModeParam1=self.getUI('label_colorModeParam1')
        self.label_colorModeParam2=self.getUI('label_colorModeParam2')
        self.label_numberShape=self.getUI('label_numberShape')
        
        self.label_normalLabel_1=self.getUI('label_normalLabel_1')
        self.label_normalLabel_2=self.getUI('label_normalLabel_2')
        self.label_normalLabel_3=self.getUI('label_normalLabel_3')
        self.label_normalLabel_4=self.getUI('label_normalLabel_4')
        self.label_normalLabel_5=self.getUI('label_normalLabel_5')
        self.label_normalLabel_6=self.getUI('label_normalLabel_6')
        self.label_normalLabel_7=self.getUI('label_normalLabel_7')

        self.collapsibleButton_PCA = self.getUI('collapsibleButton_PCA')
        self.pathLineEdit_CSVFilePCA = self.getUI('pathLineEdit_CSVFilePCA')  
        self.pathLineEdit_CSVFilePCA.filters = ctk.ctkPathLineEdit.Files
        self.pathLineEdit_CSVFilePCA.nameFilters = ['*.csv']
        self.pathLineEdit_exploration = self.getUI('pathLineEdit_exploration')
        self.pathLineEdit_exploration.filters = ctk.ctkPathLineEdit.Files
        self.pathLineEdit_exploration.nameFilters = ['*.json']
        self.comboBox_groupPCA = self.getUI('comboBox_groupPCA')
        self.comboBox_colorMode = self.getUI('comboBox_colorMode')

        self.pushButton_PCA = self.getUI('pushButton_PCA') 
        self.pushButton_resetSliders = self.getUI('pushButton_resetSliders')  
        self.pushButton_saveExploration=self.getUI('pushButton_saveExploration')
        self.pushButton_toggleMean=self.getUI('pushButton_toggleMean')
        self.pushButton_evaluateModels=self.getUI('pushButton_evaluateModels')

        self.label_statePCA = self.getUI('label_statePCA')

        self.gridLayout_PCAsliders=self.getUI('gridLayout_PCAsliders')

        self.spinBox_minVariance=self.getUI('spinBox_minVariance')
        self.spinBox_maxSlider=self.getUI('spinBox_maxSlider')
        self.spinBox_colorModeParam1=self.getUI('spinBox_colorModeParam_1')
        self.spinBox_colorModeParam2=self.getUI('spinBox_colorModeParam_2')
        self.spinBox_numberShape=self.getUI('spinBox_numberShape')
        self.spinBox_decimals = self.getUI('spinBox_decimals')
        self.label_decimals = self.getUI('label_decimals')

        self.ctkColorPickerButton_groupColor=self.getUI('ctkColorPickerButton_groupColor')

        self.checkBox_useHiddenEigenmodes=self.getUI('checkBox_useHiddenEigenmodes')

        #           Tab: PCA Export
        self.collapsibleButton_PCAExport = self.getUI('CollapsibleButton_PCAExport')
        self.comboBox_SingleExportGroup = self.getUI('comboBox_SingleExportGroup')
        self.comboBox_SingleExportPC = self.getUI('comboBox_SingleExportPC')
        self.label_PC = self.getUI('label_PC')
        self.label_Group = self.getUI('label_Group')
        self.DirectoryButton_PCASingleExport = self.getUI('DirectoryButton_PCASingleExport')
        self.pushButton_PCAExport = self.getUI('pushButton_PCAExport')
        self.pushButton_PCACurrentExport = self.getUI('pushButton_PCACurrentExport')
        self.checkBox_stdMaxMin = self.getUI('checkBox_stdMaxMin')
        self.checkBox_stdRegular = self.getUI('checkBox_stdRegular')
        self.doubleSpinBox_stdRegular = self.getUI('doubleSpinBox_stdRegular')
        self.doubleSpinBox_stdmin = self.getUI('doubleSpinBox_stdmin')
        self.doubleSpinBox_stdmax = self.getUI('doubleSpinBox_stdmax')
        self.label_stdRegular = self.getUI('label_stdRegular')
        self.label_stdmin = self.getUI('label_stdmin')
        self.label_stdmax = self.getUI('label_stdmax')
        self.doubleSpinBox_step = self.getUI('doubleSpinBox_step')

        #self.doubleSpinBox_insideLimit=self.getUI('doubleSpinBox_insideLimit')
        #self.doubleSpinBox_insideLimit=self.getUI('doubleSpinBox_outsidesideLimit')

        # --------------------------------------------------------- #
        #                  Widget Configuration                     #
        # --------------------------------------------------------- #

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
        self.spinBox_numberShape.setMinimum(100)
        self.spinBox_numberShape.setMaximum(1000000)
        self.spinBox_numberShape.setValue(10000)
        self.spinBox_decimals.setValue(3)

        self.checkBox_useHiddenEigenmodes.setChecked(True)
        
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
        self.pushButton_evaluateModels.hide()
        self.spinBox_minVariance.hide()
        self.spinBox_maxSlider.hide()
        self.label_minVariance.hide()
        self.label_maxSlider.hide()
        self.spinBox_colorModeParam1.hide()
        self.spinBox_colorModeParam2.hide()
        self.label_colorMode.hide()
        self.label_colorModeParam1.hide()
        self.label_colorModeParam2.hide()
        self.label_numberShape.hide()
        self.spinBox_numberShape.hide()
        self.spinBox_decimals.hide()
        self.label_decimals.hide()
        self.checkBox_useHiddenEigenmodes.hide()




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
        '''horizontalHeader.setResizeMode(0,qt.QHeaderView.Stretch)
        horizontalHeader.setResizeMode(1,qt.QHeaderView.ResizeToContents)
        horizontalHeader.setResizeMode(2,qt.QHeaderView.ResizeToContents)
        horizontalHeader.setResizeMode(3,qt.QHeaderView.ResizeToContents)'''
        self.tableWidget_VTKFiles.verticalHeader().setVisible(True)

        #      TAB: PCA Export
        self.pushButton_PCAExport.setEnabled(False)
        self.comboBox_SingleExportPC.setEnabled(False)
        self.comboBox_SingleExportGroup.setEnabled(False)        
        self.pushButton_PCACurrentExport.setEnabled(False)
        self.checkBox_stdRegular.setChecked(True)
        self.checkBox_stdMaxMin.setChecked(False)
        self.doubleSpinBox_stdmin.setDisabled(True)
        self.doubleSpinBox_stdmax.setDisabled(True)


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

        #          Tab: Preview / Update Classification Groups
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
        self.pushButton_evaluateModels.connect('clicked()',self.onEvaluateModels)
        self.comboBox_groupPCA.connect('activated(QString)',self.explorePCA)
        self.comboBox_colorMode.connect('activated(QString)',self.onColorModeChange)
        self.spinBox_maxSlider.connect('valueChanged(int)',self.onUpdateSliderList)
        self.spinBox_minVariance.connect('valueChanged(int)',self.onUpdateSliderList)
        self.spinBox_colorModeParam1.connect('valueChanged(int)',self.onUpdateColorModeParam)
        self.spinBox_colorModeParam2.connect('valueChanged(int)',self.onUpdateColorModeParam)
        self.ctkColorPickerButton_groupColor.connect('colorChanged(QColor)',self.onGroupColorChanged)
        self.checkBox_useHiddenEigenmodes.connect('stateChanged(int)',self.onEigenCheckBoxChanged)
        self.evaluationFlag="DONE"

        #       Tab : PCA Export
        self.pushButton_PCAExport.connect('clicked()', self.onExportForPCAExport)
        self.pushButton_PCACurrentExport.connect('clicked()', self.onExportForPCACurrentExport)
        self.checkBox_stdMaxMin.connect('clicked()', self.onMinMaxstdCheckBoxChanged)
        self.checkBox_stdRegular.connect('clicked()', self.onRegularstdCheckBoxChanged)


    def getUI(self, objectName):
        """ Functions to recovery the widget in the .ui file
        """
        return slicer.util.findChild(self.widget, objectName)

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
        #self.logic = ShapeVariationAnalyzerLogic(self)
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
        '''horizontalHeader.setResizeMode(0,qt.QHeaderView.Stretch)
        horizontalHeader.setResizeMode(1,qt.QHeaderView.ResizeToContents)
        horizontalHeader.setResizeMode(2,qt.QHeaderView.ResizeToContents)
        horizontalHeader.setResizeMode(3,qt.QHeaderView.ResizeToContents)'''
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
        self.pushButton_evaluateModels.hide()
        self.spinBox_minVariance.hide()
        self.spinBox_maxSlider.hide()
        self.label_minVariance.hide()
        self.label_maxSlider.hide()
        self.spinBox_colorModeParam1.hide()
        self.spinBox_colorModeParam2.hide()
        self.label_colorMode.hide()
        self.label_colorModeParam1.hide()
        self.label_colorModeParam2.hide()
        self.label_numberShape.hide()
        self.spinBox_numberShape.hide()
        self.spinBox_decimals.hide()
        self.label_decimals.hide()
        self.checkBox_useHiddenEigenmodes.hide()
        self.checkBox_useHiddenEigenmodes.setChecked(True)
        self.pushButton_PCA.setEnabled(False) 
        self.pathLineEdit_CSVFilePCA.disconnect('currentPathChanged(const QString)', self.onCSV_PCA)
        self.pathLineEdit_CSVFilePCA.setCurrentPath(" ")
        self.pathLineEdit_CSVFilePCA.connect('currentPathChanged(const QString)', self.onCSV_PCA)
        self.pathLineEdit_exploration.setCurrentPath(" ")

        if self.evaluationFlag!="DONE":
            self.onKillEvaluation()
        try:
            self.pushButton_evaluateModels.clicked.disconnect()
        except:
            pass
        self.pushButton_evaluateModels.setText("Evaluate models (It may take a long time)")
        self.pushButton_evaluateModels.connect('clicked()',self.onEvaluateModels)

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
        directory = self.directoryButton_creationCSVFile.directory
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
        directory = self.directoryButton_creationCSVFile.directory
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
        self.pathLineEdit_CSVFilePCA.setCurrentPath(filepath)
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
        self.fillTableForPreviewVTKFilesInSPV(self.dictVTKFiles,
                                               self.checkableComboBox_ChoiceOfGroup,
                                               self.tableWidget_VTKFiles)

        # Enable/disable buttons
        self.checkableComboBox_ChoiceOfGroup.setEnabled(True)
        self.tableWidget_VTKFiles.setEnabled(True)
        self.pushButton_previewVTKFiles.setEnabled(True)
        # self.pushButton_compute.setEnabled(True)


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
                comboBox.addItems(list(dictVTKFiles.keys()))        
                comboBox.setCurrentIndex(key)
                layout.addWidget(comboBox)
                layout.setAlignment(0x84)
                layout.setContentsMargins(0, 0, 0, 0)
                widget.setLayout(layout)
                table.setCellWidget(row, 1, widget)
                comboBox.connect('currentIndexChanged(int)', self.onGroupValueChanged)

                # Column 2:
                widget = qt.QWidget()
                layout = qt.QHBoxLayout(widget)
                checkBox = qt.QCheckBox()
                layout.addWidget(checkBox)
                layout.setAlignment(0x84)
                layout.setContentsMargins(0, 0, 0, 0)
                widget.setLayout(layout)
                table.setCellWidget(row, 2, widget)
                checkBox.connect('stateChanged(int)', self.onCheckBoxTableValueChanged)

                # Column 3:
                table.setItem(row, 3, qt.QTableWidgetItem())
                table.item(row,3).setBackground(qt.QColor(255,255,255))

                row = row + 1

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
        if os.path.exists(self.pathLineEdit_previewGroups.currentPath):
            # Creation of a color map to visualize each group with a different color in ShapePopulationViewer
            self.logic.addColorMap(self.tableWidget_VTKFiles, self.dictVTKFiles)

            # Creation of a CSV file to load the vtk files in ShapePopulationViewer
            filePathCSV = slicer.app.temporaryPath + '/' + 'VTKFilesPreview_OAIndex.csv'
            self.logic.creationCSVFileForSPV(filePathCSV, self.tableWidget_VTKFiles, self.dictVTKFiles)

            slicer.modules.shapepopulationviewer.widgetRepresentation().loadCSVFile(filePathCSV)
            slicer.util.selectModule(slicer.modules.shapepopulationviewer)

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

        try:
            self.logic.pca_exploration.loadCSVFile(self.pathLineEdit_CSVFilePCA.currentPath)
            self.pathLineEdit_exploration.setCurrentPath(" ")
        except shapca.CSVFileError as e:
            print('CSVFileError:'+e.value)
            slicer.util.errorDisplay('Invalid CSV file')
            return

        self.pushButton_PCA.setEnabled(True) 

    def onExportForExploration(self):
        self.logic.pca_exploration.process()
        self.comboBox_groupPCA.setEnabled(True)
        self.comboBox_groupPCA.clear()       

        # Activate the PCA Single Export Widgets 
        self.comboBox_SingleExportGroup.setEnabled(True)
        self.comboBox_SingleExportGroup.clear()
        self.comboBox_SingleExportPC.setEnabled(True)
        self.comboBox_SingleExportPC.clear()

        # Add personalized groups to comboboxes with the CSV
        dictPCA=self.logic.pca_exploration.getDictPCA()
        for key, value in dictPCA.items():
            group_name = value["group_name"]
            if key != "All":
                self.comboBox_groupPCA.addItem(str(key)+': '+group_name)
                self.comboBox_SingleExportGroup.addItem(str(key)+': '+group_name)
            else: 
                self.comboBox_groupPCA.addItem(key)
                self.comboBox_SingleExportGroup.addItem(key)

        self.setColorModeSpinBox()

        self.showmean=False
        self.generate3DVisualisationNodes()
        self.generate2DVisualisationNodes()

        index = self.comboBox_colorMode.findText('Group color', qt.Qt.MatchFixedString)
        if index >= 0:
             self.comboBox_colorMode.setCurrentIndex(index)

        self.pathLineEdit_exploration.disconnect('currentPathChanged(const QString)', self.onLoadExploration)
        self.pathLineEdit_exploration.setCurrentPath(' ')
        self.pathLineEdit_exploration.connect('currentPathChanged(const QString)', self.onLoadExploration)


        if self.evaluationFlag=="DONE":
            try:
                self.pushButton_evaluateModels.clicked.disconnect()
            except:
                pass
            self.pushButton_evaluateModels.setText("Evaluate models (It may take a long time)")
            self.pushButton_evaluateModels.connect('clicked()',self.onEvaluateModels)
        
        self.explorePCA()

    def onResetSliders(self):
        self.logic.pca_exploration.resetPCAPolyData()
        #self.polyDataPCA.Modified()
        for slider in self.PCA_sliders:
            slider.setSliderPosition(0)

    def onChangePCAPolyData(self, num_slider):
        ratio = self.PCA_sliders[num_slider].value

        X=1-(((ratio/1000.0)+1)/2.0)
        self.PCA_sliders_value_label[num_slider].setText(str(round(stats.norm.isf(X),3)))

        self.logic.pca_exploration.updatePolyDataExploration(num_slider,ratio/1000.0)
        #self.polyDataPCA.Modified()

    def onLoadExploration(self):

        JSONfile=self.pathLineEdit_exploration.currentPath

        # Check if the path exists:
        if not os.path.exists(JSONfile):
            return
        try:
            self.logic.pca_exploration.load(JSONfile)
            self.pathLineEdit_CSVFilePCA.disconnect('currentPathChanged(const QString)', self.onCSV_PCA)
            self.pathLineEdit_CSVFilePCA.setCurrentPath(" ")
            self.pathLineEdit_CSVFilePCA.connect('currentPathChanged(const QString)', self.onCSV_PCA)
        except shapca.JSONFileError as e:
            print('JSONFileError:'+e.value)
            slicer.util.errorDisplay('Invalid JSON file')
            return



        self.comboBox_groupPCA.setEnabled(True)
        self.comboBox_groupPCA.clear()
        dictPCA=self.logic.pca_exploration.getDictPCA()
        for key, value in dictPCA.items():

            group_name = value["group_name"]
            if key != "All":
                self.comboBox_groupPCA.addItem(str(key)+': '+group_name)
            else: 
                self.comboBox_groupPCA.addItem(key)


        self.setColorModeSpinBox()    
        self.showmean=False

        self.generate3DVisualisationNodes()
        self.generate2DVisualisationNodes()

        index = self.comboBox_colorMode.findText('Group color', qt.Qt.MatchFixedString)
        if index >= 0:
             self.comboBox_colorMode.setCurrentIndex(index)
        #slicer.mrmlScene.RemoveAllDefaultNodes()
        if self.evaluationFlag=="DONE":
            try:
                self.pushButton_evaluateModels.clicked.disconnect()
            except:
                pass
            self.pushButton_evaluateModels.setText("Evaluate models (It may take a long time)")
            self.pushButton_evaluateModels.connect('clicked()',self.onEvaluateModels)
        self.explorePCA()

    def onGroupColorChanged(self,newcolor):

        #change the plot color
        plotSeriesNode = slicer.mrmlScene.GetFirstNodeByName("PCA projection")
        plotSeriesNode.SetColor(newcolor.red()/255.0,newcolor.green()/255.0,newcolor.blue()/255.0)

        newcolor=(newcolor.red()/255.0,newcolor.green()/255.0,newcolor.blue()/255.0)
        self.logic.pca_exploration.changeCurrentGroupColor(newcolor)
        r,g,b=self.logic.pca_exploration.getColor()
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
        self.logic.pca_exploration.save(JSONpath)

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
        """ 
        Update the sliders and the 3D Visualization when the user 
        changes the Minimum Explained Variance
        """
        self.spinBox_maxSlider.value
        self.PCA_sliders
        self.PCA_sliders_label
        self.PCA_sliders_value_label

        # Extract the new number of sliders
        min_explained=self.spinBox_minVariance.value/100.0
        num_components=self.logic.pca_exploration.getRelativeNumComponent(min_explained)

        # Verify if the number of slider is not bigger than the displayable number
        if num_components>self.spinBox_maxSlider.value:
            num_components=self.spinBox_maxSlider.value

        # Change the number of sliders according to num_components
        if num_components < len(self.PCA_sliders):
            component_to_delete=len(self.PCA_sliders)-num_components
            for i in range(component_to_delete):
                self.PCA_sliders[i+num_components].deleteLater()
                self.PCA_sliders_label[i+num_components].deleteLater()
                self.PCA_sliders_value_label[i+num_components].deleteLater()
            del self.PCA_sliders[num_components : len(self.PCA_sliders)]
            del self.PCA_sliders_label[num_components : len(self.PCA_sliders_label)]
            del self.PCA_sliders_value_label[num_components : len(self.PCA_sliders_value_label)]
            self.updateVariancePlot(num_components)
            # Delete indexes in the Single Export PC combobox
            for i in range(component_to_delete,0,-1):
                self.comboBox_SingleExportPC.removeItem(i+num_components-1)

        if num_components > len(self.PCA_sliders):
            old_num_components=len(self.PCA_sliders)
            component_to_add=num_components-len(self.PCA_sliders)
            for i in range(component_to_add):
                self.createAndAddSlider(old_num_components+i, self.spinBox_decimals)
                self.comboBox_SingleExportPC.addItem(old_num_components+i+1)
            self.updateVariancePlot(num_components)

        self.logic.pca_exploration.setNumberOfVisibleEigenmodes(num_components)
        ratio = self.PCA_sliders[0].value
        self.logic.pca_exploration.updatePolyDataExploration(0,ratio/1000.0)
  
    def onColorModeChange(self):

        if self.comboBox_colorMode.currentText == 'Group color':


            self.logic.pca_exploration.setColorMode(0)
            self.spinBox_colorModeParam1.hide()
            self.spinBox_colorModeParam2.hide()
            self.label_colorModeParam1.hide()
            self.label_colorModeParam2.hide()

            self.logic.disableExplorationScalarView()

        elif self.comboBox_colorMode.currentText == 'Unsigned distance to mean shape':
            self.logic.pca_exploration.setColorModeParam(self.spinBox_colorModeParam1.value,self.spinBox_colorModeParam2.value)

            self.label_colorModeParam1.setText('Maximum Distance')
            
            self.spinBox_colorModeParam1.show()
            self.spinBox_colorModeParam2.hide()
            self.label_colorModeParam1.show()
            self.label_colorModeParam2.hide()

            self.logic.pca_exploration.setColorMode(1)

            explorationnode=slicer.mrmlScene.GetFirstNodeByName('PCA Exploration')
            colornode = slicer.mrmlScene.GetFirstNodeByName('PCA Unsigned Distance Color Table')
            if (explorationnode is not None) and (colornode is not None):

                max_dist,_=self.logic.pca_exploration.getColorParam()

                node = slicer.mrmlScene.GetFirstNodeByName("PCA Display")
                node.SetScalarRange(0,max_dist)


                explorationnode.GetDisplayNode().SetAndObserveColorNodeID(colornode.GetID())
                #explorationnode.SetInterpolate(1)
                explorationnode.Modified()



            self.logic.enableExplorationScalarView()

        elif self.comboBox_colorMode.currentText == 'Signed distance to mean shape':
            self.logic.pca_exploration.setColorModeParam(self.spinBox_colorModeParam1.value,self.spinBox_colorModeParam2.value)

            self.label_colorModeParam1.setText('Maximum Distance Outside')
            self.label_colorModeParam2.setText('Maximum Distance Inside')
            
            self.spinBox_colorModeParam1.show()
            self.spinBox_colorModeParam2.show()
            self.label_colorModeParam1.show()
            self.label_colorModeParam2.show()

            self.logic.pca_exploration.setColorMode(2)

            explorationnode=slicer.mrmlScene.GetFirstNodeByName('PCA Exploration')
            colornode = slicer.mrmlScene.GetFirstNodeByName('PCA Signed Distance Color Table')
            if (explorationnode is not None) and (colornode is not None):

                max_dist_outside,max_dist_inside=self.logic.pca_exploration.getColorParam()

                node = slicer.mrmlScene.GetFirstNodeByName("PCA Display")
                node.SetScalarRange(-max_dist_inside,max_dist_outside)

                explorationnode.GetDisplayNode().SetAndObserveColorNodeID(colornode.GetID())
                #explorationnode.SetInterpolate(1)
                explorationnode.Modified()

            self.logic.enableExplorationScalarView()

        else:
            print('Unexpected color mode option')
        return

    def onUpdateColorModeParam(self):

        self.logic.pca_exploration.setColorModeParam(self.spinBox_colorModeParam1.value,self.spinBox_colorModeParam2.value) 
        self.onColorModeChange()
        #self.logic.pca_exploration.generateDistanceColor()

    def onDataSelected(self, mrlmlPlotSeriesIds, selectionCol):
        for i in range(mrlmlPlotSeriesIds.GetNumberOfValues()):
            Id=mrlmlPlotSeriesIds.GetValue(i)
            plotserienode = slicer.mrmlScene.GetNodeByID(Id)
            if plotserienode.GetName() == "PCA projection":
                valueIds=selectionCol.GetItemAsObject(i)

                if valueIds.GetNumberOfValues()==1:
                    Id=valueIds.GetValue(0)

                #table=plotserienode.GetTableNode().GetTable()

                #pc1=table.GetValue(Id,0).ToDouble()
                #pc2=table.GetValue(Id,1).ToDouble()

                    self.logic.pca_exploration.setCurrentShapeFromId(Id)
                    self.explorePCA() 

                else:
                    Idlist=list()
                    for i in range(valueIds.GetNumberOfValues()):
                        Idlist.append(int(valueIds.GetValue(i)))
                    self.logic.pca_exploration.setCurrentShapeFromIdList(Idlist)
                    self.explorePCA() 



    def onCheckEvaluationState(self):
        state=self.evaluationThread.GetStatusString()
        if state=='Running'or state=='Scheduled':
            seconds = time.time()-self.starting_time
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            if h==0 and m==0:
                t = "00:%02d" % (s)
            elif h==0 :
                t = "%02d:%02d" % (m, s)
            else:
                t = "%d:%02d:%02d" % (h, m, s)
            if int(s) ==0:
                print("Model evaluation "+self.evaluationThread.GetStatusString()+"  "+t)
            self.pushButton_evaluateModels.setText("Abort evaluation ("+t+")")
        else:
            if self.evaluationFlag=="DONE":
                return
            print('Evaluation done')
            self.checkThreadTimer.stop()

            if self.evaluationFlag=="JSON" and self.pathLineEdit_exploration.currentPath==self.eval_param["inputJson"]:
                self.logic.pca_exploration.reloadJSONFile(self.eval_param["inputJson"])
                compactnessPCN,specificityPCN,generalizationPCN=self.generateEvaluationPlots()
                self.plotViewNode.SetPlotChartNodeID(compactnessPCN.GetID())
                self.plotViewNode.SetPlotChartNodeID(specificityPCN.GetID())
                self.plotViewNode.SetPlotChartNodeID(generalizationPCN.GetID())
                self.updateEvaluationPlots()

            if self.evaluationFlag=="CSV" and self.pathLineEdit_CSVFilePCA.currentPath==self.originalCSV:
                self.logic.pca_exploration.reloadJSONFile(self.eval_param["inputJson"])
                compactnessPCN,specificityPCN,generalizationPCN=self.generateEvaluationPlots()
                self.plotViewNode.SetPlotChartNodeID(compactnessPCN.GetID())
                self.plotViewNode.SetPlotChartNodeID(specificityPCN.GetID())
                self.plotViewNode.SetPlotChartNodeID(generalizationPCN.GetID())
                self.updateEvaluationPlots()

            self.evaluationFlag="DONE"

            self.pushButton_evaluateModels.disconnect('clicked()',self.onKillEvaluation)
            self.pushButton_evaluateModels.connect('clicked()',self.onEvaluateModels)
            self.pushButton_evaluateModels.setText("Evaluate models (It may take a long time)")

            slicer.util.infoDisplay("Evaluation done.")

    def onKillEvaluation(self):
        
        self.pushButton_evaluateModels.clicked.disconnect()
        self.pushButton_evaluateModels.connect('clicked()',self.onEvaluateModels)
        self.pushButton_evaluateModels.setText("Evaluate models (It may take a long time)")

        self.checkThreadTimer.stop()

        self.evaluationThread.Cancel()
        minutes = int((time.time()-self.starting_time)/60)
        print("Model evaluation "+self.evaluationThread.GetStatusString())
        print('Evaluation Stopped after '+str(minutes)+' min')

        self.evaluationFlag="DONE"

    def onEvaluateModels(self):
        jsonpath = self.pathLineEdit_exploration.currentPath
        csvpath = self.pathLineEdit_CSVFilePCA.currentPath
        shapeNumber=self.spinBox_numberShape.value


        if os.path.isfile(jsonpath):
            self.evaluationFlag="JSON"
            self.starting_time=time.time()

            self.eval_param = {}
            self.eval_param["inputJson"] = jsonpath
            self.eval_param["evaluation"] = str(len(self.PCA_sliders))
            self.eval_param["shapeNum"] = str(shapeNumber)

        else:
            self.originalCSV=csvpath
            self.evaluationFlag="CSV"
            self.starting_time=time.time()


            date=time.strftime("%b-%d-%Y-%H:%M:%S", time.gmtime())
            temp_dir=os.path.join(slicer.app.temporaryPath,'ShapeVariationAnalyzer_Temp')
            #temp_dir=os.path.join('/NIRAL/work/lpzmateo/data/ShapeVariationAnalyzer/','ShapeVariationAnalyzer_Temp')
            try:
                os.mkdir(temp_dir)
            except:
                pass
            temp_dir=os.path.join(temp_dir,'temp_model_'+date)
            os.mkdir(temp_dir)
            model_path=os.path.join(temp_dir,'temp.json')

            self.logic.pca_exploration.save(model_path)

            self.eval_param = {}
            self.eval_param["inputJson"] = model_path
            self.eval_param["evaluation"] = str(len(self.PCA_sliders))
            self.eval_param["shapeNum"] = str(shapeNumber)


        moduleSPCA = slicer.modules.shapepca
        self.evaluationThread=slicer.cli.run(moduleSPCA, None, self.eval_param, wait_for_completion=False)


        self.pushButton_evaluateModels.clicked.disconnect()
        self.pushButton_evaluateModels.connect('clicked()',self.onKillEvaluation)
        self.pushButton_evaluateModels.setText("Abort evaluation (00:00)")

        self.checkThreadTimer=qt.QTimer()
        self.checkThreadTimer.connect('timeout()', self.onCheckEvaluationState)
        self.checkThreadTimer.start(1000)
        return

    def onEigenCheckBoxChanged(self):
        if self.checkBox_useHiddenEigenmodes.isChecked()==True:
            self.logic.pca_exploration.useHiddenModes(True)
        else:
            self.logic.pca_exploration.useHiddenModes(False)

        ratio = self.PCA_sliders[0].value
        self.logic.pca_exploration.updatePolyDataExploration(0,ratio/1000.0)




    def explorePCA(self):
        # Detection of the selected group Id 
        if self.comboBox_groupPCA.currentText == "All":
            keygroup = "All"
        else:
            keygroup = int(self.comboBox_groupPCA.currentText[0])


        # Setting PCA model to use
        self.logic.pca_exploration.setCurrentPCAModel(keygroup)

        # Get color of the group and set the color picker with this color
        r,g,b=self.logic.pca_exploration.getColor()
        self.ctkColorPickerButton_groupColor.color=qt.QColor(int(r*255),int(g*255),int(b*255))

        # Setting the maximum number of sliders
        num_components=self.logic.pca_exploration.getNumComponent()

        if self.spinBox_maxSlider.value> num_components:
            self.spinBox_maxSlider.setMaximum(num_components)
            self.spinBox_maxSlider.setValue(num_components)
        else:
            self.spinBox_maxSlider.setMaximum(num_components)

        # Delete all the previous sliders
        self.deletePCASliders()

        # Computing the number of sliders to show
        min_explained=self.spinBox_minVariance.value/100.0
        sliders_number=self.logic.pca_exploration.getRelativeNumComponent(min_explained)
        if sliders_number>self.spinBox_maxSlider.value:
            sliders_number=self.spinBox_maxSlider.value

        # Activate the Export Buttons
        self.pushButton_PCAExport.setEnabled(True)
        self.pushButton_PCACurrentExport.setEnabled(True)

        # Create sliders and add the PC to the combobox for Single Export
        for i in range(sliders_number):
            self.createAndAddSlider(i, self.spinBox_decimals)
            self.comboBox_SingleExportPC.addItem(i+1)
       

        #Update the plot view
        self.updateVariancePlot(sliders_number)
        self.updateProjectionPlot()

        if self.logic.pca_exploration.evaluationExist():
            self.updateEvaluationPlots()


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
        self.pushButton_evaluateModels.show()
        self.spinBox_minVariance.show()
        self.spinBox_maxSlider.show()
        self.label_minVariance.show()
        self.label_maxSlider.show()

        self.label_colorMode.show()

        self.label_numberShape.show()
        self.spinBox_numberShape.show()
        self.spinBox_decimals.show()
        self.label_decimals.show()

        self.checkBox_useHiddenEigenmodes.show()



    def setColorModeSpinBox(self):
        data_std=self.logic.pca_exploration.getDataStd()
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

    def createAndAddSlider(self,num_slider, spinbox_decimals):
        exp_ratio=self.logic.pca_exploration.getExplainedRatio()
        #create the slider
        slider =qt.QSlider(qt.Qt.Horizontal)
        slider.setMaximum(999)
        slider.setMinimum(-999)
        slider.setTickInterval(1)
        position=self.logic.pca_exploration.getCurrentRatio(num_slider)
        #print(position)
        slider.setSliderPosition(position)
        #slider.setLayout(self.gridLayout_PCAsliders)
       
        #create the variance ratio label
        label = qt.QLabel()
        er = round(exp_ratio[num_slider] * 100, spinbox_decimals.value)
        if spinbox_decimals.value == 0:
            er = round(er)
        label.setText(str(num_slider+1)+':   '+str(er)+'%')
        label.setAlignment(qt.Qt.AlignCenter)

        #create the value label
        X=1-(((position/1000.0)+1)/2.0)
        '''if num_slider==4:
            print(X)'''
        valueLabel = qt.QLabel()
        valueLabel.setMinimumWidth(40)
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
        variancePCN = self.generateVariancePlot()

        projectionPCN = self.generateProjectionPlot()

        if self.logic.pca_exploration.evaluationExist():
            compactnessPCN,specificityPCN,generalizationPCN = self.generateEvaluationPlots()

        # Switch to a layout that contains a plot view to create a plot widget
        layoutManager = slicer.app.layoutManager()
        layoutManager.setLayout(36)

        # Select chart in plot view
        plotWidget = layoutManager.plotWidget(0)
        self.plotViewNode = plotWidget.mrmlPlotViewNode()


        if self.logic.pca_exploration.evaluationExist():
            self.plotViewNode.SetPlotChartNodeID(compactnessPCN.GetID())
            self.plotViewNode.SetPlotChartNodeID(specificityPCN.GetID())
            self.plotViewNode.SetPlotChartNodeID(generalizationPCN.GetID())

        self.plotViewNode.SetPlotChartNodeID(projectionPCN.GetID())
        self.plotViewNode.SetPlotChartNodeID(variancePCN.GetID())

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

        self.deleteFunctionPlot("Generalization")
        self.deleteFunctionPlot("Specificity")
        self.deleteFunctionPlot("Compactness")
        
    def deleteFunctionPlot(self,name_y):
        node = slicer.mrmlScene.GetFirstNodeByName("PCA "+name_y+" table")
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)
        node = slicer.mrmlScene.GetFirstNodeByName(name_y)
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)
        node = slicer.mrmlScene.GetFirstNodeByName("PCA "+name_y+" plot chart")
        if node is not None:
            slicer.mrmlScene.RemoveNode(node)


    def generateProjectionPlot(self):
        projectionTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode","PCA projection table")
        table = projectionTableNode.GetTable()

        pc1=vtk.vtkFloatArray()
        pc2=vtk.vtkFloatArray()
        labels = vtk.vtkStringArray()

        pc1.SetName("pc1")
        pc2.SetName("pc2")
        labels.SetName("files")

        table.AddColumn(pc1)
        table.AddColumn(pc2)
        table.AddColumn(labels)

        #Projection plot serie
        projectionPlotSeries = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "PCA projection")
        projectionPlotSeries.SetAndObserveTableNodeID(projectionTableNode.GetID())
        projectionPlotSeries.SetXColumnName("pc1")
        projectionPlotSeries.SetYColumnName("pc2")
        projectionPlotSeries.SetLabelColumnName("files")
        projectionPlotSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        projectionPlotSeries.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleNone)
        projectionPlotSeries.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleSquare)        
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

    def generateEvaluationPlots(self):
        #compactness
        compactnessPCN=self.generateFunctionPlot("Component","Compactness")
        #specificity
        specificityPCN=self.generateFunctionPlot("Component","Specificity")
        #generalization
        generalizationPCN=self.generateFunctionPlot("Component","Generalization")

        return compactnessPCN,specificityPCN,generalizationPCN

    def generateFunctionPlot(self,name_x,name_y):
        self.deleteFunctionPlot(name_y)

        TableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode","PCA "+name_y+" table")
        table = TableNode.GetTable()

        x=vtk.vtkFloatArray()
        y=vtk.vtkFloatArray()

        x.SetName(name_x)
        y.SetName(name_y)

        table.AddColumn(x)
        table.AddColumn(y)

        #Sum Explained specificity plot serie
        PlotSeries = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", name_y)
        PlotSeries.SetAndObserveTableNodeID(TableNode.GetID())
        PlotSeries.SetXColumnName(name_x)
        PlotSeries.SetYColumnName(name_y)
        PlotSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        PlotSeries.SetUniqueColor()

        # Create specificity plot chart node
        PCN = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode","PCA "+name_y+" plot chart")
        PCN.AddAndObservePlotSeriesNodeID(PlotSeries.GetID())
        PCN.SetTitle(name_y)
        PCN.SetXAxisTitle(name_x)
        PCN.SetYAxisTitle(name_y) 

        return PCN


    def updateVariancePlot(self,num_components):

        varianceTableNode = slicer.mrmlScene.GetFirstNodeByName("PCA variance table")
        table = varianceTableNode.GetTable()
        table.Initialize()

        level95 , level1=self.logic.pca_exploration.getPlotLevel(num_components)
        level95.SetName("level95%")
        level1.SetName("level1%")
        table.AddColumn(level95)
        table.AddColumn(level1)

        x,evr,sumevr= self.logic.pca_exploration.getPCAVarianceExplainedRatio(num_components)
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

        pc1,pc2=self.logic.pca_exploration.getPCAProjections(normalized=True)
        labels = self.logic.pca_exploration.getPCAProjectionLabels()

        pc1.SetName("pc1")
        pc2.SetName("pc2")
        labels.SetName("files")

        table.AddColumn(pc1)
        table.AddColumn(pc2)
        table.AddColumn(labels)

        #update color
        plotSeriesNode = slicer.mrmlScene.GetFirstNodeByName("PCA projection")
        r, g, b = self.logic.pca_exploration.getColor()
        plotSeriesNode.SetColor(r, g, b)

        #fit to contents
        layoutManager = slicer.app.layoutManager()
        plotWidget = layoutManager.plotWidget(0)
        plotWidget.plotController().fitPlotToAxes()


    def updateEvaluationPlots(self):   
        #compactness
        x,compac,compac_err=self.logic.pca_exploration.getCompactness()
        self.updateFunctionPlot("Component","Compactness",x,compac)

        #specificity
        x,spec,spec_err=self.logic.pca_exploration.getSpecificity()
        self.updateFunctionPlot("Component","Specificity",x,spec)

        #generalization
        x,gene,gene_err=self.logic.pca_exploration.getGeneralization()
        self.updateFunctionPlot("Component","Generalization",x,gene)



    def updateFunctionPlot(self,name_x,name_y,x,y):
        TableNode = slicer.mrmlScene.GetFirstNodeByName("PCA "+name_y+" table")
        table = TableNode.GetTable()
        table.Initialize()

        x.SetName(name_x)
        y.SetName(name_y)

        table.AddColumn(x)
        table.AddColumn(y)


    #polydata
    def generate3DVisualisationNodes(self):
        self.delete3DVisualisationNodes()
        ##For Mean shape
        #clear scene from previous PCA exploration
        
        #create Model Node
        PCANode = slicer.vtkMRMLModelNode()
        PCANode.SetAndObservePolyData(self.logic.pca_exploration.getPolyDataMean())
        PCANode.SetName("PCA Mean")
        #create display node
        
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
        PCANode.SetAndObservePolyData(self.logic.pca_exploration.getPolyDataExploration())
        PCANode.SetName("PCA Exploration")
        #create display node
        R,G,B=self.logic.pca_exploration.getColor()
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


    # ----------------------------------------------------  #
    #                                                       #
    #                   Tab: PCA Export                     #
    #                                                       #
    # ----------------------------------------------------  #

    def onMinMaxstdCheckBoxChanged(self):
        if self.checkBox_stdMaxMin.isChecked()==True:
            self.checkBox_stdRegular.setChecked(False)
            self.doubleSpinBox_stdRegular.setDisabled(True)
            self.doubleSpinBox_stdmin.setDisabled(False)
            self.doubleSpinBox_stdmax.setDisabled(False)
        else:
            self.checkBox_stdMaxMin.setChecked(True)


    def onRegularstdCheckBoxChanged(self):
        if self.checkBox_stdRegular.isChecked()==True:
            self.checkBox_stdMaxMin.setChecked(False)
            self.doubleSpinBox_stdRegular.setDisabled(False)
            self.doubleSpinBox_stdmin.setDisabled(True)
            self.doubleSpinBox_stdmax.setDisabled(True)
        else:
            self.checkBox_stdRegular.setChecked(True)

    def onExportForPCAExport(self):
        """ Function to export the CSV file in the directory chosen by the user
            - Save the CSV file from the dictionary previously filled
            - Load automatically this CSV file in the next tab: "Creation of New Classification Groups"
        """
        # Path of the vtk file

        #dlg = ctk.ctkFileDialog()
        #filepath = dlg.getSaveFileName(None, "Export CSV file for Classification groups", os.path.join(qt.QDir.homePath(), "Desktop"), "CSV File (*.csv)")
        
        # Variables for the file's name
        Group = self.comboBox_SingleExportGroup.itemText(self.comboBox_SingleExportGroup.currentIndex)
        PC = self.comboBox_SingleExportPC.itemText(self.comboBox_SingleExportPC.currentIndex)
        std_regular = self.doubleSpinBox_stdRegular.textFromValue(self.doubleSpinBox_stdRegular.value)
        step = self.doubleSpinBox_step.value



        # X=1-(((ratio/1000.0)+1)/2.0) So we want the inverse to have ration/1000.0

        num_slider = int(PC)-1

        position_Slider=self.logic.pca_exploration.getCurrentRatio(num_slider)
        self.onResetSliders()

        ratio = 1000*((1- stats.norm.sf(float(std_regular)))*2 - 1)
        # print("Ratio: " , ratio , " Ancienne position: ", position_Slider )

        # Creation of a folder
        folder_int = 1
        while (os.path.exists(self.DirectoryButton_PCASingleExport.directory + '/PCAMultipleAxisExport_' + str(folder_int) + '/')):
            folder_int += 1
        Folder = '/PCAMultipleAxisExport_' + str(folder_int) + '/'

        # Begining of the filepath
        dirpath = self.DirectoryButton_PCASingleExport.directory + Folder

        # Creation of the folder for the different deviations
        try:
            os.mkdir(dirpath)
        except OSError:
            print ("Creation of the directory %s failed" % dirpath)
        else:
            print ("Successfully created the directory %s " % dirpath)

        # CREATION OF THE VTK FILES
        if (self.checkBox_stdRegular.isChecked()):    
            std_max = float(std_regular)
            std_min = -float(std_regular)
        else:
            std_max = self.doubleSpinBox_stdmax.value
            std_min = self.doubleSpinBox_stdmin.value\
        # From the mean, we go first to the max and then from the mean to the min (non-symetrical range)
        for std_count in np.arange(0.0,std_max,step):
            self.logic.exportAxis(dirpath,Group,PC,std_count)
        for std_count in np.arange(-step,std_min,-step):
            self.logic.exportAxis(dirpath,Group,PC,std_count)
        # The two limits are also exported
        self.logic.exportAxis(dirpath,Group,PC,std_max)
        self.logic.exportAxis(dirpath,Group,PC,std_min)
        # The previous visualisation is set back
        self.PCA_sliders[num_slider].setSliderPosition(position_Slider)
            


    def onExportForPCACurrentExport(self):        
        """ Export the current vistualisation of the module in a vtk file """
        Group = "0"
        PC = "1"
        std = "5.9"
        file_number = 1

        dirpath = self.DirectoryButton_PCASingleExport.directory + "/PCACurrentExport/"
        if( os.path.exists(dirpath)==False ):
            # Creation of the folder for the current explorations if it's not existing
            try:
                os.mkdir(dirpath)
            except OSError:
                print ("Creation of the directory %s failed" % dirpath)
            else:
                print ("Successfully created the directory %s " % dirpath)

        # Display the sign
        if (std[0]!='-'):
            std = '+' + std
        filepath_current = dirpath + "/PCA_Group" + Group[0] + "_MixedComp_" + str(file_number)
        #To don't overwrite on the other current exports
        exist = False
        while (exist == False):
            #If the file already exists we just increment the last number
            if (os.path.exists(filepath_current + ".vtk")):
                filepath_end = len(filepath_current) - len(str(file_number))
                file_number += 1
                filepath_current = filepath_current[:filepath_end] + str(file_number)
            else:
                exist = True
        filepath_current = filepath_current + ".vtk"

        self.logic.pca_exploration.saveVTKFile(self.logic.pca_exploration.getPolyDataExploration(),filepath_current) 



# ------------------------------------------------------------------------------------------ #
#                                                                                            #
#                                                                                            #
#                                    LOGIC                                                   #
#                                                                                            #
#                                                                                            #
# ------------------------------------------------------------------------------------------ #

class ShapeVariationAnalyzerLogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        self.table = vtk.vtkTable
        self.colorBar = {'Point1': [0, 0, 1, 0], 'Point2': [0.5, 1, 1, 0], 'Point3': [1, 1, 0, 0]}

        self.pca_exploration=shapca.pcaExplorer()
        

    def addGroupToDictionary(self, dictCSVFile, directory, directoryList, group):
        """ Function to add all the vtk filepaths 
        found in the given directory of a dictionary
        """
        # Fill a dictionary which contains the vtk files for the classification groups sorted by group
        valueList = list()
        for file in os.listdir(directory):
            if file.endswith(".vtk") or file.endswith(".xml"):
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
                if isinstance(value, list):
                    value.append(self.table.GetValue(i,0).ToString())
                else:
                    tempList = list()
                    tempList.append(value)
                    tempList.append(self.table.GetValue(i,0).ToString())
                    dict[self.table.GetValue(i,1).ToInt()] = tempList

        return True

    def addColorMap(self, table, dictVTKFiles):
        """ Function to add a color map "DisplayClassificationGroup" 
        to all the vtk files which allow the user to visualize each 
        group with a different color in ShapePopulationViewer
        """
        for key, value in dictVTKFiles.items():
            for vtkFile in value:
                polyData = vtk.vtkPolyData()
                if os.path.basename(vtkFile).endswith(".vtk"):
                    # Read VTK File
                    reader = vtk.vtkDataSetReader()
                    reader.SetFileName(vtkFile)
                    reader.ReadAllVectorsOn()
                    reader.ReadAllScalarsOn()
                    reader.Update()
                    polyData = reader.GetOutput()
                elif os.path.basename(vtkFile).endswith(".xml"):
                    # Read S-Rep file
                    parser = vtk.vtkXMLDataParser()
                    parser.SetFileName(vtkFile)
                    parser.Parse()
                    root = parser.GetRootElement()

                    reader0 = vtk.vtkXMLPolyDataReader()
                    reader0.SetFileName(root.FindNestedElementWithName("upSpoke").GetCharacterData())
                    reader0.Update()

                    reader1 = vtk.vtkXMLPolyDataReader()
                    reader1.SetFileName(root.FindNestedElementWithName("downSpoke").GetCharacterData())
                    reader1.Update()

                    reader2 = vtk.vtkXMLPolyDataReader()
                    reader2.SetFileName(root.FindNestedElementWithName("crestSpoke").GetCharacterData())
                    reader2.Update()

                    append = vtk.vtkAppendPolyData()
                    append.AddInputData(CPNS.extractSpokes(reader0.GetOutput(), 0))
                    append.AddInputData(CPNS.extractSpokes(reader1.GetOutput(), 4))
                    append.AddInputData(CPNS.extractSpokes(reader2.GetOutput(), 2))
                    append.AddInputData(CPNS.extractEdges(reader0.GetOutput(), 1))
                    append.AddInputData(CPNS.extractEdges(reader2.GetOutput(), 3))
                    append.Update()
                    polyData = append.GetOutput()

                    vtkFile = os.path.splitext(vtkFile)[0]+'.vtk'
                else:
                    print("Wrong file type!")

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
                if os.path.exists(filepath):
                    print("Error: two input files under different groups have the same name!")
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
                if vtkFile.endswith('.xml'):
                    vtkFile = os.path.splitext(vtkFile)[0] + '.vtk'
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
                comboBox.addItems(list(dictVTKFiles.keys()))        
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
        
    def removeDataVTKFiles(self, value):
        """ Function to remove in the temporary directory all 
        the data used to create the mean for each group
        """
        # remove of all the vtk file
        for vtkFile in value:
            if vtkFile.endswith('.xml'):
                vtkFile = os.path.splitext(vtkFile)[0] + '.vtk'
            filepath = slicer.app.temporaryPath + '/' + os.path.basename(vtkFile)
            if os.path.exists(filepath):
                os.remove(filepath)

    def creationCSVFile(self, directory, CSVbasename, dictForCSV, option):
        """ Function to create a CSV file:
            - Two columns are always created:
                - First column: path of the vtk files
                - Second column: group associated to this vtk file
            - If saveH5 is True, this CSV file will contain a New Classification Group, a thrid column is then added
                - Third column: path of the shape model of each group
        """
        CSVFilePath = str(directory) + "/" + CSVbasename
        file = open(CSVFilePath, 'w')
        cw = csv.writer(file, delimiter=',')
        if option == "Groups":
            cw.writerow(['VTK Files', 'Group'])
        elif option == "MeanGroup":
            cw.writerow(['Mean shapes VTK Files', 'Group'])
        for key, value in dictForCSV.items():
            if isinstance(value, list):
                for vtkFile in value:
                    if option == "Groups":
                        cw.writerow([vtkFile, str(key)])
                
            elif option == "MeanGroup":
                cw.writerow([value, str(key)])

            elif option == "NCG":
                cw.writerow([value, str(key)])
        file.close()

    def checkSeveralMeshInDict(self, dict):
        """ Function to check if in each group 
        there is at least more than one mesh
        """
        for key, value in dict.items():
            if type(value) is not type(list()) or len(value) == 1:
                msg='The group ' + str(key) + ' must contain more than one mesh.'
                raise shapca.CSVFileError(msg)
                return False
        return True

    #################
    # PCA ALGORITHM #       
    #################

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

    def exportAxis(self,dirpath,Group,PC,std_count):
        num_slider = int(PC)-1
        filepath = dirpath + "PCA_Group" + Group[0] + "_Comp" + PC + "_std" + str(round(std_count,2)) + ".vtk"
        ratio = 1000*((1- stats.norm.sf(std_count))*2 - 1)
        self.pca_exploration.updatePolyDataExploration(num_slider,ratio/1000.0)
        self.pca_exploration.saveVTKFile(self.pca_exploration.getPolyDataExploration(),filepath)



class ShapeVariationAnalyzerTest(ScriptedLoadableModuleTest):
    """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def setUp(self):
        pass

    def runTest(self):
        self.setUp()
        self.delayDisplay('Starting the tests')
        self.test_ShapeVariationAnalyzer()

    def test_ShapeVariationAnalyzer(self):
        self.logic = ShapeVariationAnalyzerLogic()
        filepath_in = "./hippo.csv"
        # Test of all the groups
        keygroup = "All"

        try:
            self.logic.pca_exploration.loadCSVFile(filepath_in)
            self.logic.pca_exploration.process()
            # Add personalized groups to comboboxes with the CSV
            dictPCA = self.logic.pca_exploration.getDictPCA()
            # Setting PCA model to use
            self.logic.pca_exploration.setCurrentPCAModel(keygroup)
        except shapca.CSVFileError as e:
            print('CSVFileError:'+e.value)
            slicer.util.errorDisplay('Invalid CSV file')
    
        try:
            exp_ratio=self.logic.pca_exploration.getExplainedRatio()
            error_bool = False
            # Values for the hippo PCA exploration
            comparison = [38.075,9.688,6.970,5.525,4.338,3.643,2.835,2.487]
            for num_slider in range(8):
                #print ( str(comparison[num_slider]) + " compare to " + str(round(exp_ratio[num_slider]*100,3)) )
                if ( comparison[num_slider] != round(exp_ratio[num_slider]*100,3) ):
                    error_bool = True
            if (error_bool == True):
                print( 'Exploration Error: The PCA results are wrong.')
            else:
                print( "The PCA exploration is right.")
        except:
            print( 'Exploration failed' )
            slicer.util.errorDisplay('Exploration failed')

        filepath_out = "./test.vtk"
        #Export Curent visualisation
        self.logic.pca_exploration.saveVTKFile(self.logic.pca_exploration.getPolyDataExploration(),filepath_out) 

        if (os.path.exists(filepath_out)):
            print(filepath_out + " created.")
        else:
            print('CSVExportError: '+filepath_out + " not created.")
            slicer.util.errorDisplay('Target not created')

        os.remove(filepath_out)

        if (os.path.exists(filepath_out) == False):
            print(filepath_out + " deleted.")
        else:
            print('CSVExportError: '+filepath_out + " not deleted.")
            slicer.util.errorDisplay('Target not deleted')
