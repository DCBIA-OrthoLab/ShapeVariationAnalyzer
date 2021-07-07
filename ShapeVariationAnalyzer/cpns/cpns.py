import numpy as np
import os
import sys
import vtk

from xml.dom import minidom
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from .principal_nested_spheres import PNS
from vtk.util.numpy_support import vtk_to_numpy as vtk_to_numpy


class CPNS(object):
    """
    Fit composite principle nested spheres model to skeletal representations as described in:
    Pizer S.M. et al. (2013) Nested Sphere Statistics of Skeletal Models.
    In: Breuß M., Bruckstein A., Maragos P. (eds) Innovations for Shape Analysis. Mathematics and Visualization.
    Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-34141-0_5

    NOTE:
        (1) PCA is perform outside the class by using the transformed composite Z matrix(self.ZComp) and
        shape can be reconstructed by using self.getPolyData(CPNSScores, polyData)
        (2) The input shape will be re-centered using mean hub point positions.

    Author: Ye Han
    Date: Jun 30, 2021
    """

    def __init__(self):

        # Files
        self.fileList = []

        # CPNS info
        self.nRows = None
        self.nCols = None
        self.nSamples = None
        self.nTotalAtoms = None
        self.nEndAtoms = None
        self.nStdAtoms = None
        self.nTotalPositions = None
        self.nTotalRadii = None

        # derived Data
        self.upSpokes = []
        self.downSpokes = []
        self.crestSpokes = []
        self.pointMap = None
        self.meanSizePDM = None
        self.meanOfCombinedPDM = None
        self.meanRs = None
        self.PNSShape = None
        self.PNSSpoke = None
        self.ZComp = None

    def Update(self):
        if len(self.fileList) is 0:
            print("No input file list! Please use setInputFiles(fileList)")
            self.clear()
            return False

        for filePath in self.fileList:
            # Check extension
            if os.path.basename(filePath).endswith(".xml") is False:
                print("Wrong file format in the list!")
                self.clear()
                return False
            dirName = os.path.dirname(filePath)

            # Read S-Rep file
            parser = vtk.vtkXMLDataParser()
            parser.SetFileName(filePath)
            parser.Parse()
            root = parser.GetRootElement()

            # Check number of rows and columns
            nRows = int(root.FindNestedElementWithName("nRows").GetCharacterData())
            nCols = int(root.FindNestedElementWithName("nCols").GetCharacterData())
            if self.nRows is None or self.nCols is None:
                self.nRows = nRows
                self.nCols = nCols
            elif nRows is not self.nRows or nCols is not self.nCols:
                print("nRows/nCols does not match!")
                self.clear()
                return False

            # Add .vtp files
            reader0 = vtk.vtkXMLPolyDataReader()
            upSpokeName = root.FindNestedElementWithName("upSpoke").GetCharacterData()
            if os.path.exists(upSpokeName) is False:  # if using relative path
                upSpokeName = dirName + "/" + upSpokeName
            reader0.SetFileName(upSpokeName)
            reader0.Update()
            self.upSpokes.append(reader0.GetOutput())

            reader1 = vtk.vtkXMLPolyDataReader()
            downSpokeName = root.FindNestedElementWithName("downSpoke").GetCharacterData()
            if os.path.exists(downSpokeName) is False:  # if using relative path
                downSpokeName = dirName + "/" + downSpokeName
            reader1.SetFileName(downSpokeName)
            reader1.Update()
            self.downSpokes.append(reader1.GetOutput())

            reader2 = vtk.vtkXMLPolyDataReader()
            crestSpokeName = root.FindNestedElementWithName("crestSpoke").GetCharacterData()
            if os.path.exists(crestSpokeName) is False:  # if using relative path
                crestSpokeName = dirName + "/" + crestSpokeName
            reader2.SetFileName(crestSpokeName)
            reader2.Update()
            self.crestSpokes.append(reader2.GetOutput())

        # Get dataset info
        self.nSamples = len(self.fileList)
        self.nTotalAtoms = self.upSpokes[0].GetNumberOfPoints()
        self.nEndAtoms = self.crestSpokes[0].GetNumberOfPoints()
        self.nStdAtoms = self.nTotalAtoms - self.nEndAtoms
        self.nTotalPositions = 3 * (self.nEndAtoms + self.nStdAtoms)
        self.nTotalRadii = 3 * self.nEndAtoms + 2 * self.nStdAtoms

        self.pointMap = np.zeros(self.crestSpokes[0].GetNumberOfPoints())
        for i in range(self.crestSpokes[0].GetNumberOfPoints()):
            distance = sys.float_info.max
            pt0 = np.array(self.crestSpokes[0].GetPoint(i))
            for j in range(self.upSpokes[0].GetNumberOfPoints()):
                pt1 = np.array(self.upSpokes[0].GetPoint(j))
                newDistance = np.linalg.norm(pt1 - pt0)
                if newDistance < distance:
                    distance = newDistance
                    self.pointMap[i] = j

        # CPNS: Step 1 : Deal with hub Positions (PDM)
        position = np.zeros((self.nTotalAtoms, 3, self.nSamples))
        for i in range(self.nSamples):
            for j in range(self.nTotalAtoms):
                position[j, :, i] = self.upSpokes[i].GetPoint(j)

        meanOfEachPDM = np.mean(position, 0, keepdims=True)
        self.meanOfCombinedPDM = np.mean(meanOfEachPDM, 2, keepdims=False)
        cposition = position - np.repeat(meanOfEachPDM, self.nTotalAtoms, 0)
        sscarryPDM = np.sqrt(np.sum(np.sum(cposition ** 2, 0), 0))
        sphmatPDM = np.zeros((self.nTotalPositions, self.nSamples))
        spharrayPDM = np.zeros((self.nTotalAtoms, 3, self.nSamples))
        for i in range(self.nSamples):
            spharrayPDM[:, :, i] = cposition[:, :, i] / sscarryPDM[i]
            sphmatPDM[:, i] = np.reshape(spharrayPDM[:, :, i], (self.nTotalAtoms * 3))

        # Fit PNS to data
        pnsModel = PNS(sphmatPDM, itype=2)  # TODO: set as an optional parameter?
        pnsModel.fit()
        ZShape, self.PNSShape = pnsModel.output

        sizePDM = np.zeros((1, self.nSamples))
        sizePDM[0, :] = sscarryPDM
        self.meanSizePDM = np.exp(np.mean(np.log(sizePDM), keepdims=True))
        normalizedSizePDM = np.log(sizePDM / self.meanSizePDM)

        # CPNS: Step 2 : Deal with atom radii (log radii are Euclidean variables)
        logR = np.zeros((self.nTotalRadii, self.nSamples))
        for i in range(self.nSamples):
            upR = vtk_to_numpy(self.upSpokes[i].GetPointData().GetScalars('spokeLength'))
            downR = vtk_to_numpy(self.downSpokes[i].GetPointData().GetScalars('spokeLength'))
            crestR = vtk_to_numpy(self.crestSpokes[i].GetPointData().GetScalars('spokeLength'))
            logR[:, i] = np.log(np.concatenate((upR, downR, crestR)))

        meanLogR = np.mean(logR, axis=1, keepdims=True)
        self.meanRs = np.exp(meanLogR)
        rScaleFactors = np.repeat(self.meanRs, self.nSamples, axis=1)

        uScaleFactors = np.zeros((2 * self.nTotalRadii, self.nSamples))
        for i in range(self.nTotalRadii):
            uScaleFactors[2 * i, :] = rScaleFactors[i, :]
            uScaleFactors[2 * i + 1, :] = rScaleFactors[i, :]

        RStar = logR - np.repeat(meanLogR, self.nSamples, axis=1)

        # CPNS: Step 3 : Deal with spoke directions (direction analysis)
        spokeDirections = np.zeros((3 * self.nTotalRadii, self.nSamples))
        for i in range(self.nSamples):
            upD = np.reshape(vtk_to_numpy(self.upSpokes[i].GetPointData().GetArray('spokeDirection')), -1)
            downD = np.reshape(vtk_to_numpy(self.downSpokes[i].GetPointData().GetArray('spokeDirection')), -1)
            crestD = np.reshape(vtk_to_numpy(self.crestSpokes[i].GetPointData().GetArray('spokeDirection')), -1)
            spokeDirections[:, i] = np.concatenate((upD, downD, crestD))

        ZSpoke = np.zeros((2 * self.nTotalRadii, self.nSamples))
        self.PNSSpoke = []
        for i in range(self.nTotalRadii):
            pnsModel = PNS(spokeDirections[(3 * i):(3 * i + 3), :], itype=9)
            pnsModel.fit()
            zD, pnsD = pnsModel.output
            self.PNSSpoke.append(pnsD)
            ZSpoke[(2 * i):(2 * i + 2), :] = zD

        # CPNS: Step 4 : Construct composite Z matrix
        self.ZComp = np.concatenate((np.multiply(self.meanSizePDM, ZShape),
                                     np.multiply(self.meanSizePDM, normalizedSizePDM),
                                     np.multiply(rScaleFactors, RStar),
                                     np.multiply(uScaleFactors, ZSpoke)),
                                    axis=0)

        # CPNS: Step 5 : Perform PCA outside

    def setInputFileList(self, fileList):
        self.fileList = fileList

    def clear(self):
        # CPNS info
        self.nRows = None
        self.nCols = None
        self.nSamples = None
        self.nTotalAtoms = None
        self.nEndAtoms = None
        self.nStdAtoms = None
        self.nTotalPositions = None
        self.nTotalRadii = None

        # derived Data
        self.upSpokes = []
        self.downSpokes = []
        self.crestSpokes = []
        self.pointMap = None
        self.meanSizePDM = None
        self.meanOfCombinedPDM = None
        self.meanRs = None
        self.PNSShape = None
        self.PNSSpoke = None
        self.ZComp = None

    def getPolyData(self, CPNSScores):

        X, radii, spokeDirs = self.computeShape(CPNSScores)
        upSpoke = self.createUpSpokePolyData(X, radii, spokeDirs)
        downSpoke = self.createDownSpokePolyData(X, radii, spokeDirs)
        crestSpoke = self.createCrestSpokePolyData(X, radii, spokeDirs)

        append = vtk.vtkAppendPolyData()
        append.AddInputData(self.extractSpokes(upSpoke, 0))
        append.AddInputData(self.extractSpokes(downSpoke, 4))
        append.AddInputData(self.extractSpokes(crestSpoke, 2))
        append.AddInputData(self.extractEdges(upSpoke, 1))
        append.AddInputData(self.extractEdges(crestSpoke, 3))
        append.Update()
        polyData = append.GetOutput()
        return polyData

    def writeSRep(self, CPNSScores, filePath):

        if not os.path.basename(filePath).endswith('.xml'):
            print("Error: filepath should end with .xml")
            return
        fileName = os.path.splitext(filePath)[0]

        X, radii, spokeDirs = self.computeShape(CPNSScores)
        upSpoke = self.createUpSpokePolyData(X, radii, spokeDirs)
        downSpoke = self.createDownSpokePolyData(X, radii, spokeDirs)
        crestSpoke = self.createCrestSpokePolyData(X, radii, spokeDirs)

        # Write .xml file
        # borrowed from SlicerSkeletalRepresentation/SRepVisualizer/LegacyTransformer/legacyTransformer.py
        root = Element('s-rep')
        nRowsXMLElement = SubElement(root, 'nRows')
        nRowsXMLElement.text = str(self.nRows)
        nColsXMLElement = SubElement(root, 'nCols')
        nColsXMLElement.text = str(self.nCols)
        meshTypeXMLElement = SubElement(root, 'meshType')
        meshTypeXMLElement.text = 'Quad'
        colorXMLElement = SubElement(root, 'color')
        redXMLElement = SubElement(colorXMLElement, 'red')
        redXMLElement.text = str(0)
        greenXMLElement = SubElement(colorXMLElement, 'green')
        greenXMLElement.text = str(0.5)
        blueXMLElement = SubElement(colorXMLElement, 'blue')
        blueXMLElement.text = str(0)
        isMeanFlagXMLElement = SubElement(root, 'isMean')
        isMeanFlagXMLElement.text = 'False'
        meanStatPathXMLElement = SubElement(root, 'meanStatPath')
        meanStatPathXMLElement.text = ''
        # later this can be done as a parameter
        upSpokeXMLElement = SubElement(root, 'upSpoke')
        upSpokeXMLElement.text = './' + os.path.basename(fileName) + '_up.vtp'  # Use relative path
        downSpokeXMLElement = SubElement(root, 'downSpoke')
        downSpokeXMLElement.text = './' + os.path.basename(fileName) + '_down.vtp'
        crestSpokeXMLElement = SubElement(root, 'crestSpoke')
        crestSpokeXMLElement.text = './' + os.path.basename(fileName) + '_crest.vtp'
        crestShiftXMLElement = SubElement(root, 'crestShift')
        crestShiftXMLElement.text = '0.0'
        file_handle = open(filePath, 'w')
        file_handle.write(self.prettify(root))
        file_handle.close()

        # write .vtp files
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(fileName + '_up.vtp')
        writer.SetInputData(upSpoke)
        writer.Write()

        writer.SetFileName(fileName + '_down.vtp')
        writer.SetInputData(downSpoke)
        writer.Write()

        writer.SetFileName(fileName + '_crest.vtp')
        writer.SetInputData(crestSpoke)
        writer.Write()

    """
    borrowed from https://pymotw.com/2/xml/etree/ElementTree/create.html
    """
    def prettify(self, elem):
        """
        :param elem:
        :return: a pretty-printed XML string for the Element
        """
        roughString = ElementTree.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(roughString)
        return reparsed.toprettyxml(indent="  ")

    def computeShape(self, CPNSScores):
        # Check dimension
        CPNSDim = CPNSScores.shape[0]
        if CPNSDim is not self.ZComp.shape[0]:
            print("Error, input CPNS scores dimension is incorrect!")
            return None

        # Construct hub positions
        if CPNSDim < self.nTotalPositions + 3 * self.nTotalRadii:
            dimZShape = CPNSDim - 3 * self.nTotalRadii
        else:
            dimZShape = self.nTotalPositions
        zShape = CPNSScores[0:dimZShape - 1] / self.meanSizePDM
        zSizePDM = CPNSScores[dimZShape - 1]
        sizePDMOverall = np.multiply(self.meanSizePDM, np.exp(zSizePDM / self.meanSizePDM))[0, 0]
        XStar = PNS.inv(zShape, self.PNSShape)[:, 0]
        X = np.add(sizePDMOverall * XStar, np.tile(self.meanOfCombinedPDM[0, :], self.nTotalAtoms))

        # Construct radii
        zRStar = CPNSScores[dimZShape: dimZShape + self.nTotalRadii]
        radii = np.multiply(self.meanRs, np.exp(np.divide(zRStar, self.meanRs)))

        # Construct spokeDirs
        zSpokes = CPNSScores[dimZShape + self.nTotalRadii: dimZShape + 3 * self.nTotalRadii]
        uScaleFactors = np.zeros((2 * self.nTotalRadii, 1))
        for i in range(self.nTotalRadii):
            uScaleFactors[2 * i, :] = self.meanRs[i]
            uScaleFactors[2 * i + 1, :] = self.meanRs[i]
        zSpokes = np.divide(zSpokes, uScaleFactors)
        spokeDirs = np.zeros((3 * self.nTotalRadii, 1))
        for ns in range(self.nTotalRadii):
            spokeDir = PNS.inv(zSpokes[2 * ns:(2 * ns + 2), :], self.PNSSpoke[ns])
            spokeDirs[3 * ns:(3 * ns + 3), [0]] = spokeDir

        return X, radii, spokeDirs

    @staticmethod
    def projectShape(self, filePath):
        # TODO: project an input shape onto the feature space for statistical analysis on unseen data
        pass

    def createUpSpokePolyData(self, X, radii, spokeDirs):
        upSpoke = vtk.vtkPolyData()
        upSpoke.DeepCopy(self.upSpokes[0])
        for i in range(self.nTotalAtoms):
            upSpoke.GetPoints().SetPoint(i, X[3 * i:3 * i + 3])
            upSpoke.GetPointData().GetScalars('spokeLength').SetTuple1(i, radii[i])
            upSpoke.GetPointData().GetScalars('spokeDirection').SetTuple3(i,
                                                                          spokeDirs[3 * i],
                                                                          spokeDirs[3 * i + 1],
                                                                          spokeDirs[3 * i + 2])
        return upSpoke

    def createDownSpokePolyData(self, X, radii, spokeDirs):
        downSpoke = vtk.vtkPolyData()
        downSpoke.DeepCopy(self.downSpokes[0])
        for i in range(self.nTotalAtoms):
            downSpoke.GetPoints().SetPoint(i, X[3 * i:3 * i + 3])
            downSpoke.GetPointData().GetScalars('spokeLength').SetTuple1(i, radii[i + self.nTotalAtoms])
            downSpoke.GetPointData().GetScalars('spokeDirection').SetTuple3(i,
                                                                            spokeDirs[3 * (i + self.nTotalAtoms)],
                                                                            spokeDirs[3 * (i + self.nTotalAtoms) + 1],
                                                                            spokeDirs[3 * (i + self.nTotalAtoms) + 2])
        return downSpoke

    def createCrestSpokePolyData(self, X, radii, spokeDirs):
        crestSpoke = vtk.vtkPolyData()
        crestSpoke.DeepCopy(self.crestSpokes[0])
        for i in range(crestSpoke.GetNumberOfPoints()):
            crestSpoke.GetPoints().SetPoint(i, X[int(3 * self.pointMap[i]):int(3 * self.pointMap[i] + 3)])
            crestSpoke.GetPointData().GetScalars('spokeLength').SetTuple1(i, radii[i + 2 * self.nTotalAtoms])
            crestSpoke.GetPointData().GetScalars('spokeDirection').SetTuple3(i,
                                                                             spokeDirs[3 * (i + 2 * self.nTotalAtoms)],
                                                                             spokeDirs[3 * (i + 2 * self.nTotalAtoms) + 1],
                                                                             spokeDirs[3 * (i + 2 * self.nTotalAtoms) + 2])
        return crestSpoke

    @staticmethod
    def extractSpokes(polyData, cellType):

        spokePoints = vtk.vtkPoints()
        spokeLines = vtk.vtkCellArray()
        typeArray = vtk.vtkIntArray()
        typeArray.SetName("cellType")
        typeArray.SetNumberOfComponents(1)
        arr_length = polyData.GetPointData().GetArray("spokeLength")
        arr_dirs = polyData.GetPointData().GetArray("spokeDirection")

        for i in range(polyData.GetNumberOfPoints()):
            pt0 = polyData.GetPoint(i)
            spokeLength = arr_length.GetTuple1(i)
            direction = arr_dirs.GetTuple3(i)
            pt1 = [pt0[0] + spokeLength * direction[0],
                   pt0[1] + spokeLength * direction[1],
                   pt0[2] + spokeLength * direction[2]]
            id0 = spokePoints.InsertNextPoint(pt0)
            id1 = spokePoints.InsertNextPoint(pt1)

            arrow = vtk.vtkLine()
            arrow.GetPointIds().SetId(0, id0)
            arrow.GetPointIds().SetId(1, id1)
            spokeLines.InsertNextCell(arrow)
            typeArray.InsertNextValue(cellType)
            typeArray.InsertNextValue(cellType)

        spokePD = vtk.vtkPolyData()
        spokePD.SetPoints(spokePoints)
        spokePD.SetLines(spokeLines)
        spokePD.GetPointData().SetActiveScalars("cellType")
        spokePD.GetPointData().SetScalars(typeArray)

        return spokePD

    @staticmethod
    def extractEdges(polyData, cellType):
        edgeExtractor = vtk.vtkExtractEdges()
        edgeExtractor.SetInputData(polyData)
        edgeExtractor.Update()
        edges = edgeExtractor.GetOutput()

        outputType = vtk.vtkIntArray()
        outputType.SetName("cellType")
        outputType.SetNumberOfComponents(1)

        for i in range(edges.GetNumberOfPoints()):
            outputType.InsertNextValue(cellType)
        edges.GetPointData().SetActiveScalars("cellType")
        edges.GetPointData().SetScalars(outputType)
        return edges

    def getZCompMatrix(self):
        return self.ZComp
