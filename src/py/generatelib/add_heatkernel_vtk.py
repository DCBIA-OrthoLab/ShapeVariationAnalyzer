import argparse
import inputData
import glob
import os
import vtk
import scipy.io as sio
import numpy as np
from vtk.util import numpy_support


parser = argparse.ArgumentParser(description='Shape Variation Analyzer')
parser.add_argument('--dataPath', action='store', dest='dirwithSub', help='folder with subclasses', required=True)


if __name__ == '__main__':
	
	args = parser.parse_args()
	dataPath=args.dirwithSub

	inputdata = inputData.inputData()
	data_folders = inputdata.get_folder_classes_list(dataPath)
	
	print(data_folders)
	polydata = vtk.vtkPolyData()

	for datafolders in data_folders:
		i=0
		vtklist = glob.glob(os.path.join(datafolders, "*.vtk"))
		print(vtklist)
		matfile = glob.glob(os.path.join(datafolders,"*.mat"))
		matfile_str = ''.join(map(str,matfile))

		print ('matfile',matfile_str)
		for matlabfilename in matfile:
			mat_contents = sio.loadmat(matlabfilename,squeeze_me=True)
			shape_matlab = mat_contents['shape']
			for vtkfilename in vtklist:
			#if vtkfilename.endswith((".vtk")):
			#	print i
				if(vtkfilename[:-4] == matlabfilename[:-4]):
				#vtkfilename = matfile_str.replace('.mat','.vtk')
					print('vtkfilename',vtkfilename)
					reader = vtk.vtkPolyDataReader()
					reader.SetFileName(vtkfilename)
					reader.Update()
					polydata = reader.GetOutput()
				#for matfilename in matlist:
					#if matfilename.endswith((".mat")):
					print('################')
					#shape1 = shape_matlab[i]
					heat_kernel = np.array(shape_matlab['sihks'].tolist())
					print (heat_kernel.shape)
					for j in range(0,19):
						print(j)
						heat_kernel_2 = heat_kernel[:,j]
						shape_heat_kernel = heat_kernel_2.shape
						print(shape_heat_kernel)
						heat_kernel_data = numpy_support.numpy_to_vtk(heat_kernel_2.ravel(),deep=True,array_type=vtk.VTK_FLOAT)



						heat_kernel_data.SetNumberOfComponents(1);
						heat_kernel_data.SetName('heat_kernel_signature_'+str(j));
						
						polydata.GetPointData().AddArray(heat_kernel_data)					

						print("Writing", vtkfilename)
						writer = vtk.vtkPolyDataWriter()
						writer.SetFileName(vtkfilename)
						writer.SetInputData(polydata)
						writer.Write()
					
						i+=1;

