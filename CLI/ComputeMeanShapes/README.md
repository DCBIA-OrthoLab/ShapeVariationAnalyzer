# ComputeMeanShapes

## Description

From a list of VTK 3D models, this program computes the average shape. 
Each point of the output model is the resulting average of the corresponding across the input list of shapes.

## Build

#### Requirements 

Building ComputeMeanShape requires to have built previously:

* VTK version 7
* ITK 
* SlicerExecutionModel
* CMake


###### Linux or MacOSX 

Consider this tree of repertories:
```
~/Project/ComputeMeanShape
         /ComputeMeanShape-build
```

Start a terminal.
First change your current working directory to the build directory ```ComputeMeanShape-build```
```
cd ~/Project/ComputeMeanShape-build
```

Generate the project using ```cmake```
```
cmake -DVTK_DIR:PATH=path/to/VTK -DITK_DIR:PATH=path/to/ITK -DSlicerExecutionModel:PATH=path/to/SlicerExecutionModel ../ComputeMeanShape
make
```


## Usage

```
./computemean [--inputList <std::vector<std::string>>] [--outputSurface <std::string>]
```


## Licence

See LICENSE.txt for information on using and contributing.