#ifndef COMPUTEMEAN_H
#define COMPUTEMEAN_H

#include <vtkSmartPointer.h>
#include <vtkPolyDataAlgorithm.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>

#include <vtkSmartPointer.h>
#include <vtkPolyDataAlgorithm.h>
#include <vtkPolyDataNormals.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>

#include "fileIO.h"


class ComputeMean : public vtkPolyDataAlgorithm
{
public:
    /** Conventions for a VTK Class*/
    vtkTypeMacro(ComputeMean,vtkPolyDataAlgorithm);
    static ComputeMean *New(); 

    /** Function SetInput(std::string input, std::vector<std::string> list)
    * Set the inputs data of the filter
    * @param input : input shape
    * @param list : list of group mean shapes
    */
    void SetInput(std::vector<std::string> input, int num);

    /** Function Update()
     * Update the filter and process the output
     */
    void Update();


    void compute_mean_shape();

    /**
     * Return the output of the Filter
     * @return : output of the Filter ComputeMean
     */
    vtkSmartPointer<vtkPolyData> GetOutput();

private:
    /** Variables */
    std::vector< vtkSmartPointer<vtkPolyData> > inputSurfaces;

    vtkSmartPointer<vtkPolyData> outputSurface;

    vtkSmartPointer<vtkPolyData> intermediateSurface;
    
    int number;

    /** Function init_output()
     * Initialize outputSurface
     */
    void init_output();




protected:
    /** Constructor & Destructor */
    ComputeMean();
    ~ComputeMean();

};

#endif // COMPUTEMEAN_H