import argparse
import inputData
import glob
import os
import vtk
import scipy.io as sio
import numpy as np
from vtk.util import numpy_support

#from pyevtk.hl import imageToVTK


parser = argparse.ArgumentParser(description='Shape Variation Analyzer')

parser.add_argument('--dataPath', action='store', dest='dirwithSub', help='folder with subclasses', required=True)
#parser.add_argument('--matlabfile', action='store', dest='matlab', help='matlab file with', required=True)

#parser.add_argument('--vtkfile', action='store', dest='vtkfile', help='vtkfile to complete with heat_kernel', required=True)
#parser.add_argument('--matlabfile', action='store', dest='matlab', help='matlab file with heat kernel signature', required=True)


if __name__ == '__main__':
	
	args = parser.parse_args()
	dataPath=args.dirwithSub
	#matFile =args.matlab
	#vtkfile = args.vtkfile

	inputdata = inputData.inputData()
	data_folders = inputdata.get_folder_classes_list(dataPath)
	
	print(data_folders)
	polydata = vtk.vtkPolyData()
	
	#print(shape_matlab[104])
	#i=1

	for datafolders in data_folders:
		i=0
		vtklist = glob.glob(os.path.join(datafolders, "*.vtk"))
		print(vtklist)
		matfile = glob.glob(os.path.join(datafolders,"*.mat"))
		matfile_str = ''.join(map(str,matfile))

		print ('matfile',matfile)
		#print(vtklist)
		print('h')
		mat_contents = sio.loadmat(matfile_str,squeeze_me=True)
		shape_matlab = mat_contents['shape']
		for vtkfilename in vtklist:
			if vtkfilename.endswith((".vtk")):
				print(i)
				reader = vtk.vtkPolyDataReader()
				reader.SetFileName(vtkfilename)
				reader.Update()
				polydata = reader.GetOutput()
		#for matfilename in matlist:
			#if matfilename.endswith((".mat")):
				print('################')
				shape1 = shape_matlab[i]
				heat_kernel = np.array(shape1['sihks'].tolist())
				print (heat_kernel.shape)
				for j in range(0,19):
					print(j)
					heat_kernel_2 = heat_kernel[:,j]
					#H = heat_kernel * 1
					shape_heat_kernel = heat_kernel_2.shape
					print(shape_heat_kernel)
					#VTK_data = numpy_support.numpy_to_vtk(num_array=H.ravel(),deep=True,array_type=vtk.VTK_INT)
					heat_kernel_data = numpy_support.numpy_to_vtk(heat_kernel_2.ravel(),deep=True,array_type=vtk.VTK_FLOAT)


					#creation polydata object

					#i_max = shape_heat_kernel[0]
					#j_max = shape_heat_kernel[1]
					heat_kernel_data.SetNumberOfComponents(1);
					heat_kernel_data.SetName('heat_kernel_signature_'+str(j));
				
				#if i==104:
				#	i=0;


		#polydata.SetPoints(heat_kernel_data)
	#for i in range(0,i_max):
		#for j in range(1,j_max):
			#print heat_kernel[i,j]
			#heat_kernel_data.InsertNextValue(heat_kernel[i])
		#print i
	
	#print heat_kernel_data
	#print('**************')

	#print('hello')
					polydata.GetPointData().AddArray(heat_kernel_data)
		#print('point',polydata.GetPointData())
		#print(polydata.GetPointData().GetArray("heat_kernel_signature"))


						

					print("Writing", vtkfilename)
					writer = vtk.vtkPolyDataWriter()
					writer.SetFileName(vtkfilename)
					writer.SetInputData(polydata)
					writer.Write()
			
				i+=1;

