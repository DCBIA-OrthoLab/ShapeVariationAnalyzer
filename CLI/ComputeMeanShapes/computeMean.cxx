#include <iostream>

#include "computeMeanCLP.h"
#include "computeMean.hxx"

#include <iterator>
#include <algorithm>


int main(int argc, char** argv)
{
    PARSE_ARGS;

    vtkSmartPointer<ComputeMean> Filter = vtkSmartPointer<ComputeMean>::New();
    vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();

    if(argc < 3)
    {
        std::cout << "Usage " << argv[0] << " [--inputList <std::vector<std::string>>] [--outputSurface <std::string>] " << std::endl;
        return 1;
    }

    // Check input dir & Get shapes list
    std::vector<std::string> listShapes;
    for (auto it = inputList.begin(); it != inputList.end(); ++it)
    {
        (*it).erase(std::remove((*it).begin(), (*it).end(), ' '), (*it).end());
        (*it).erase(std::remove((*it).begin(), (*it).end(), '\''), (*it).end());
        listShapes.push_back(*it);
    }   
    // Check output file
    if(outputSurface.rfind(".vtk")==std::string::npos || outputSurface.empty())
    {
        std::cerr << "Wrong Output File Format, must be a .vtk file" << std::endl;
        return EXIT_FAILURE;
    }

    int groupNumber = 0;
    Filter->SetInput(listShapes, groupNumber);
    Filter->Update();

    writer->SetFileName(outputSurface.c_str());
    writer->SetInputData(Filter->GetOutput());
    writer->Update();

    return EXIT_SUCCESS;
}

