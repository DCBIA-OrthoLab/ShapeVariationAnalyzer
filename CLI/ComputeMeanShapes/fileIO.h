#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkXMLPolyDataReader.h>
#include <itkMacro.h> //For itkException

/**
 * Read a VTK File
 * @param   - Location of the file to read
 * @return  - Data of the file
 */
vtkSmartPointer<vtkPolyData> readVTKFile (std::string filename);

/**
 * Write in a VTK Format (.vtp or .vtk) a file at the location thrown in parameter
 * @param filename - Location of the file to write
 * @param output   - Data of the file to write
 */
void writeVTKFile (std::string filename, vtkSmartPointer<vtkPolyData> output);