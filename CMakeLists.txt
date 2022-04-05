cmake_minimum_required(VERSION 3.13.4)

project(ShapeVariationAnalyzer)

set(ShapeVariationAnalyzer_VERSION_MAJOR 1)
set(ShapeVariationAnalyzer_VERSION_MINOR 0)
set(ShapeVariationAnalyzer_VERSION_PATCH 4)


#-----------------------------------------------------------------------------
# Standalone vs Slicer extension option
#-----------------------------------------------------------------------------

# This option should be named after the project name, it corresponds to the
# option set to ON when the project is build by the Slicer Extension build
# system.

set(_default OFF)
set(_reason "${PROJECT_NAME}_BUILD_SLICER_EXTENSION is ON")
if(NOT DEFINED ${PROJECT_NAME}_BUILD_SLICER_EXTENSION AND DEFINED Slicer_DIR)
  set(_default ON)
  set(_reason "Slicer_DIR is SET")
endif()

option(${PROJECT_NAME}_BUILD_SLICER_EXTENSION "Build as a Slicer Extension" ${_default})

set(_msg "Checking if building as a Slicer extension")
message(STATUS ${_msg})
if(${PROJECT_NAME}_BUILD_SLICER_EXTENSION)
  message(STATUS "${_msg} - yes (${_reason})")
else()
  message(STATUS "${_msg} - no (${PROJECT_NAME}_BUILD_SLICER_EXTENSION is OFF)")
endif()

if(ShapeVariationAnalyzer_BUILD_SLICER_EXTENSION)
  #-----------------------------------------------------------------------------
  # Extension meta-information
  set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/ShapeVariationAnalyzer")
  set(EXTENSION_CATEGORY "Quantification")
  set(EXTENSION_CONTRIBUTORS "Priscille de Dumast (University of Michigan), Laura Pascal (University of Michigan), Nina Tubau (University of Michigan), Juan Carlos Prieto (University of North Carolina) ")
  set(EXTENSION_DESCRIPTION "ShapeVariationAnalyzer allows the classification of 3D models, according to their morphological variations. This tool is based on a deep learning neural network.")
  set(EXTENSION_ICONURL "https://www.slicer.org/w/images/3/30/ShapeVariationAnalyzer-Logo.png")
  set(EXTENSION_SCREENSHOTURLS "https://www.slicer.org/w/images/f/f0/FullView_ShapeVariationAnalyzer.png https://www.slicer.org/w/images/5/57/ShapeVariationAnalyzer-GUI.png")
  set(EXTENSION_DEPENDS ShapePopulationViewer)
  set(EXTENSION_BUILD_SUBDIRECTORY inner-build)

  set(SUPERBUILD_TOPLEVEL_PROJECT inner)

  #-----------------------------------------------------------------------------
  # Extension dependencies
  find_package(Slicer REQUIRED)
  include(${Slicer_USE_FILE})
  mark_as_superbuild(Slicer_DIR)

  find_package(Git REQUIRED)
  mark_as_superbuild(GIT_EXECUTABLE)

  #-----------------------------------------------------------------------------
  # SuperBuild setup
  option(${EXTENSION_NAME}_SUPERBUILD "Build ${EXTENSION_NAME} and the projects it depends on." ON)
  mark_as_advanced(${EXTENSION_NAME}_SUPERBUILD)
  if(${EXTENSION_NAME}_SUPERBUILD)
    include("${CMAKE_CURRENT_SOURCE_DIR}/SuperBuild.cmake")
    return()
  endif()

  #-----------------------------------------------------------------------------
  # Extension modules
  add_subdirectory(ShapeVariationAnalyzer)
else()
  find_package(VTK REQUIRED)
  find_package(SlicerExecutionModel REQUIRED)
endif()

#-----------------------------------------------------------------------------
# Extension modules

#add_subdirectory(CLI/ComputeMeanShapes)
add_subdirectory(CLI/SurfaceFeaturesExtractor)
## NEXT_MODULE


#-----------------------------------------------------------------------------
if(ShapeVariationAnalyzer_BUILD_SLICER_EXTENSION)

  if(NOT Slicer_SOURCE_DIR)
    #-----------------------------------------------------------------------------
    # Install extension python packages
    install(CODE "message(\"CPack: - Install directory: ${python_packages_DIR}\")")
    install(
        DIRECTORY "${python_packages_DIR}/"
        DESTINATION ${Slicer_INSTALL_ROOT}${Slicer_BUNDLE_EXTENSIONS_LOCATION}
        COMPONENT RuntimeLibraries
        )
  endif()

  #-----------------------------------------------------------------------------
  set(EXTENSION_CPACK_INSTALL_CMAKE_PROJECTS)
  set(${EXTENSION_NAME}_CPACK_INSTALL_CMAKE_PROJECTS "${EXTENSION_CPACK_INSTALL_CMAKE_PROJECTS}" CACHE STRING "List of external projects to install" FORCE)

  #-----------------------------------------------------------------------------
  list(APPEND CPACK_INSTALL_CMAKE_PROJECTS "${CMAKE_BINARY_DIR};${EXTENSION_NAME};ALL;/")
  list(APPEND CPACK_INSTALL_CMAKE_PROJECTS "${${EXTENSION_NAME}_CPACK_INSTALL_CMAKE_PROJECTS}")
  include(${Slicer_EXTENSION_GENERATE_CONFIG})
  include(${Slicer_EXTENSION_CPACK})
endif()
