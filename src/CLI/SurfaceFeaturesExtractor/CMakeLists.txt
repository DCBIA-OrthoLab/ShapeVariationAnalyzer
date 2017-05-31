

cmake_minimum_required(VERSION 2.8)

PROJECT(SurfaceFeaturesExtractor)

#Find VTK Package
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

#Find ITK Package
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

#Find Slicer Execution Model Package
find_package(SlicerExecutionModel REQUIRED)
include(${SlicerExecutionModel_USE_FILE})


# Add repertory src/
include_directories(src)
add_subdirectory(src)







