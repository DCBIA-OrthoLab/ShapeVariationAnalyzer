# Surface Features Extractor

## Description

Specific features are computed at each point of the 3D Model. Those features are added to the vtk file as scalar, vector or array.

Default features: 
  
  * Normals (Vector)
  * Positions (3 scalars)
  * Maximum curvature (1 scalar)
  * Minimum curvature (1 scalar)
  * Gaussian curvature (1 scalar)
  * Mean curvature (1 scalar)
  * Shape Index (1 scalar)
  * Curvedness (1 scalar)

Optional features:

  * landmarks (1 array)
  * distances to other mesh (as many scalar as distMesh provided)

## Build

#### Requirements 

Building SurfaceFeaturesExtractor requires to have built previously:

* VTK version 7
* ITK 
* SlicerExecutionModel
* CMake


###### Linux or MacOSX 

Consider this tree of repertories:
```
~/Project/SurfaceFeaturesExtractor
         /SurfaceFeaturesExtractor-build
```

Start a terminal.
First change your current working directory to the build directory ```SurfaceFeaturesExtractor-build```
```
cd ~/Project/SurfaceFeaturesExtractor-build
```

Generate the project using ```cmake```
```
cmake -DVTK_DIR:PATH=path/to/VTK -DITK_DIR:PATH=path/to/ITK -DSlicerExecutionModel:PATH=path/to/SlicerExecutionModel ../SurfaceFeaturesExtractor
make
```


## Usage

```
./surfacefeaturesextractor inputModel outputModel 
[options: 
--distMeshOn --distMesh [<std::vector<std::string>> list of vtk files]
--landmarksOn --landmarks [<std::string> fcsv file] ]
```


## Licence

See LICENSE.txt for information on using and contributing.