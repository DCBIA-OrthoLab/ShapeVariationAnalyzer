#include "fileIO.h"


vtkSmartPointer<vtkPolyData> readVTKFile (std::string filename)
{           // VTK
	if (filename.rfind(".vtk") != std::string::npos)
	{
        // std::cout<<"---Reading VTK input file at "<<filename.c_str()<<std::endl;
        vtkSmartPointer<vtkPolyDataReader> fiberReader = vtkPolyDataReader::New();
        fiberReader->SetFileName(filename.c_str());
        if(fiberReader->OpenVTKFile())
        {
            fiberReader->Update();
            return fiberReader->GetOutput();
        }
        else
        {
            throw itk::ExceptionObject("File Non Valid");
        }

    }
	        // XML
    else if (filename.rfind(".vtp") != std::string::npos)
    {
        // std::cout<<"---Reading VTP input file at "<<filename.c_str()<<std::endl;
        vtkSmartPointer<vtkXMLPolyDataReader> fiberReader = vtkXMLPolyDataReader::New();
        fiberReader->SetFileName(filename.c_str());
        if(fiberReader->CanReadFile(filename.c_str()))
        {
            fiberReader->Update();
            return fiberReader->GetOutput();
        }
        else
        {
            throw itk::ExceptionObject("File Non Valid");
        }
    }
    else
    {
        throw itk::ExceptionObject("Unknown file format for fibers");
    }

}

void writeVTKFile (std::string filename, vtkSmartPointer<vtkPolyData> output)
{	
	if (filename.rfind(".vtk") != std::string::npos)
	{
        // std::cout<<"---Writing VTK output File to "<<filename.c_str()<<std::endl;
        vtkSmartPointer<vtkPolyDataWriter> fiberWriter = vtkPolyDataWriter::New();

        fiberWriter->SetFileName(filename.c_str());

		// #if (VTK_MAJOR_VERSION < 6)
            // fiberWriter->SetInput(output);
		// #else
            fiberWriter->SetInputData(output);
		// #endif
            fiberWriter->Update();
    }
	        // XML
    else if (filename.rfind(".vtp") != std::string::npos)
    {
        // std::cout<<"---Writting VTP output File to "<<filename.c_str()<<std::endl;
    	vtkSmartPointer<vtkXMLPolyDataWriter> fiberWriter = vtkXMLPolyDataWriter::New();
    	fiberWriter->SetFileName(filename.c_str());
		// #if (VTK_MAJOR_VERSION < 6)
        	// fiberWriter->SetInput(output);
		// #else
        	fiberWriter->SetInputData(output);
		// #endif
        	fiberWriter->Update();
    }
    else
    {
        throw itk::ExceptionObject("Unknown file format for fibers");
    }

	
}