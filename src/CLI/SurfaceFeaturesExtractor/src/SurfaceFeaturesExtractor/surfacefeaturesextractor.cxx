#include "surfacefeaturesextractor.hxx"
#include "fileIO.hxx"
#include "surfacefeaturesextractorCLP.h"

#include <iterator>


int main (int argc, char *argv[])
{
	PARSE_ARGS;
    
    vtkSmartPointer<vtkPolyData> inputShape = readVTKFile(inputMesh.c_str());
    int num_points = inputShape->GetNumberOfPoints();

    std::vector< vtkSmartPointer<vtkPolyData> > distMeshList;
    if ( distMeshOn )
    {
        // Load each mesh used for distances 
        for (int k=0; k<distMesh.size(); k++) 
        {
            vtkSmartPointer<vtkPolyData> crt_mesh = vtkSmartPointer<vtkPolyData>::New();
            crt_mesh = readVTKFile( distMesh[k].c_str() );

            if (crt_mesh->GetNumberOfPoints() != num_points)
            {
                std::cerr << "All the shapes must have the same number of points" << std::endl;
                return EXIT_FAILURE;
            } 
            distMeshList.push_back(crt_mesh);
        }
    }

    std::vector< std::string> landmarkFile;
    if ( landmarksOn )
        landmarkFile.push_back(landmarks);

    vtkSmartPointer<SurfaceFeaturesExtractor> Filter = vtkSmartPointer<SurfaceFeaturesExtractor>::New();
	Filter->SetInput(inputShape, distMeshList, landmarkFile);
	Filter->Update();
    writeVTKFile(outputMesh.c_str(),Filter->GetOutput());

	return EXIT_SUCCESS;
}
