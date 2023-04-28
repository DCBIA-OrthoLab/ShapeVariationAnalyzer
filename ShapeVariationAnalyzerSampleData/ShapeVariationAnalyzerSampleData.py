import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
from pathlib import Path
import shutil

#
# ShapeVariationAnalyzerSampleData
#

class ShapeVariationAnalyzerSampleData(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "ShapeVariationAnalyzerSampleData"
    self.parent.categories = ["TestData"]
    self.parent.dependencies = ["SampleData"]
    self.parent.helpText = """This module adds sample data for ShapeVariationAnalyzer"""

    # don't show this module - additional data will be shown in SampleData module
    parent.hidden = True 

    import SampleData    
    
    iconsPath = os.path.join(os.path.dirname(self.parent.path), 'Resources/Icons/')
    
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
      sampleName='ShapeVariationAnalyzerSampleData',
      category='ShapeVariationAnalyzer',
      uris='https://data.kitware.com/api/v1/item/626850ed4acac99f42121c7f/download',
      loadFiles=True, # Unzip it
      fileNames='ShapeVariationAnalyzerSampleData.zip',
      loadFileType='ZipFile',
      nodeNames='ShapeVariationAnalyzerSampleData',
      checksums='SHA512:c2a8b8b33b18bb9ad6308afd2c557a7743edaf77f3e03d50f4df0059d6aa4cdd78093dda144ed2f0a27f088464e30d6e654fc68cd19b37b6027e44bebf97bfc3',
      thumbnailFileName=os.path.join(iconsPath, 'bendicon.png'),
      customDownloader=self.downloadSampleDataInFolder,
    )
    
  @staticmethod
  def downloadSampleDataInFolder(source):
  
    outPath = slicer.mrmlScene.GetCacheManager().GetRemoteCacheDirectory()
        
    if not os.path.exists(outPath):
      os.mkdir(outPath)
    
    sampleDataLogic = slicer.modules.sampledata.widgetRepresentation().self().logic
    
    for uri, fileName, checksums in zip(source.uris, source.fileNames, source.checksums):
      sampleDataLogic.downloadFile(uri, outPath, fileName)
    
