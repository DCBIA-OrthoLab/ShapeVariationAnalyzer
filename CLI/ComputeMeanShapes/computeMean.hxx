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
	std::vector<std::string>::iterator it = input.begin(), it_end = input.end();
    for (; it != it_end; it++)
        this->inputSurfaces.push_back(readVTKFile((*it).c_str()));

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
			delete[] p;
		}
	
		sum[0] = sum[0] / nbSurf;
		sum[1] = sum[1] / nbSurf;
		sum[2] = sum[2] / nbSurf;

		this->outputSurface->GetPoints()->SetPoint(i, sum);

		delete[] sum;

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


