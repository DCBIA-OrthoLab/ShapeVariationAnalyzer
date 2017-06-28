#include "computeMean.h"



vtkStandardNewMacro(ComputeMean);

/**
* Constructor ComputeMean()
*/
ComputeMean::ComputeMean(){
	this->inputSurfaces = std::vector < vtkSmartPointer<vtkPolyData> > () ;
	this->outputSurface = vtkSmartPointer<vtkPolyData>::New();
	this->number = 0;
}

/**
* Destructor ComputeMean()
*/
ComputeMean::~ComputeMean(){}

/**
* Function SetInput() for ComputeMean
*/
void ComputeMean::SetInput(std::vector<std::string> input, int num)
{
    vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
	std::string fileName;
	std::vector<std::string>::iterator it = input.begin(), it_end = input.end();
    for (; it != it_end; it++)
    {
    	fileName = *it;
    	std::cout<<"---Reading VTK input file at "<<fileName.c_str()<<std::endl;
        reader->SetFileName(fileName.c_str());
        reader->Update();
        this->inputSurfaces.push_back(reader->GetOutput());
    }
    this->number = num;
}

/**
 * Function init_output() for ComputeMean
 */
void ComputeMean::init_output()
{
	this->outputSurface = this->inputSurfaces[0];
}

void ComputeMean::compute_mean_shape()
{
	std::cout<<"---Starting computation"<<std::endl;
	int nbPoints = this->inputSurfaces[0]->GetNumberOfPoints();
	for (int i=0; i< nbPoints; i++)
	{
	    double* sum = new double[3];
	    sum[0] = 0; sum[1] = 0; sum[2] = 0; 

	    int nbSurf = this->inputSurfaces.size();
	    std::vector< vtkSmartPointer<vtkPolyData> >::iterator it = this->inputSurfaces.begin(), it_end = this->inputSurfaces.end();
	    for (; it != it_end; it++)
		{
			double* p = new double[3];
			p = (*it)->GetPoint(i);

			sum[0] = sum[0] + p[0];
			sum[1] = sum[1] + p[1];
			sum[2] = sum[2] + p[2]; 
		}
	
		sum[0] = sum[0] / nbSurf;
		sum[1] = sum[1] / nbSurf;
		sum[2] = sum[2] / nbSurf;

		this->outputSurface->GetPoints()->SetPoint(i, sum);

	}

}


/**
 * Function Update()
 */
void ComputeMean::Update()
{
	this->init_output();

	// Compute mean shape
	this->compute_mean_shape();
	puts("Mean shape computed");
}

/**
 * Function GetOutput()
 */
vtkSmartPointer<vtkPolyData> ComputeMean::GetOutput()
{
	return this->outputSurface;
}


