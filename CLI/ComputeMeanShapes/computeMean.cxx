#include <iostream>

#include "computeMeanCLP.h"
#include "computeMean.hxx"
#include "fileIO.hxx"

#include <iterator>


int main(int argc, char** argv)
{
    PARSE_ARGS;

    vtkSmartPointer<ComputeMean> Filter = vtkSmartPointer<ComputeMean>::New();

    if(argc < 3)
    {
        std::cout << "Usage " << argv[0] << " [--inputList <std::vector<std::string>>] [--outputSurface <std::string>] " << std::endl;
        return 1;
    }

    // Check input dir & Get shapes list
    std::vector<std::string> listShapes;
    listShapes = inputList;
       
    // Check output file
    if(outputSurface.rfind(".vtk")==std::string::npos || outputSurface.empty())
    {
        std::cerr << "Wrong Output File Format, must be a .vtk file" << std::endl;
        return EXIT_FAILURE;
    }

    int groupNumber = 0;
    Filter->SetInput(listShapes, groupNumber);
    Filter->Update();
    writeVTKFile(outputSurface.c_str(),Filter->GetOutput());

    return EXIT_SUCCESS;
}

