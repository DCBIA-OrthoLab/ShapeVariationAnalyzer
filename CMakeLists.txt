cmake_minimum_required(VERSION 2.8.9)

project(ShapeVarationAnalyzer)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/ShapeVarationAnalyzer")
set(EXTENSION_CATEGORY "Quantification")
set(EXTENSION_CONTRIBUTORS "Laura Pascal (University of Michigan), Priscille de Dumast (University of Michigan)")
set(EXTENSION_DESCRIPTION "Shape Variation Analyzer allows the classification of 3D models, according to their morphological variations. This tool is based on a deep learning neural network.")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/ShapeVarationAnalyzer.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/ShapeVarationAnalyzer/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any
# set(EXTENSION_DEPENDS ShapePopulationViewer)

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(src)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
