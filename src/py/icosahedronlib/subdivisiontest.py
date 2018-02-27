import vtk
import numpy as np 

#Coords in unit sphere for icosahedron

def get_subdiv_points(icosahedron, subdivisionlevel):
	points = icosahedron.GetPoints()
	for cellid in range(icosahedron.GetNumberOfCells()):
		idlist = vtk.vtkIdList()
		icosahedron.GetCellPoints(cellid, idlist)
		
		idlist.GetId(0)
		idlist.GetId(1)
		idlist.GetId()


def main():

	colors = vtk.vtkNamedColors()

	bkg = map(lambda x: x / 256.0, [51, 77, 102])
	# bkg = map(lambda x: x / 256.0, [26, 51, 77])
	colors.SetColor("BkgColor", *bkg)

	sourceObjects = list()

	subdivisionlevel = 6

	icosahedronsource = vtk.vtkPlatonicSolidSource()
	icosahedronsource.SetSolidTypeToIcosahedron()
	icosahedronsource.Update()

	sourceObjects.append(icosahedronsource.GetOutput())
	icosahedron = sourceObjects[-1]

	for sl in range(1, subdivisionlevel):
		subdivpoints = get_subdiv_points(icosahedron, sl)

		surfrecon = vtk.vtkSurfaceReconstructionFilter()
		surfrecon.SetInputData(subdivpoints)

		# subdivfilter = vtk.vtkLinearSubdivisionFilter()
		# subdivfilter.SetInputData(icosahedron)
		# subdivfilter.SetNumberOfSubdivisions(sl)
		# subdivfilter.Update()
		
		# subdivpoly = subdivfilter.GetOutput()
		# subdivpoints = subdivpoly.GetPoints()

		# for pid in range(subdivpoints.GetNumberOfPoints()):
		# 	spoint = subdivpoints.GetPoint(pid)
		# 	spoint = np.array(spoint)
		# 	norm = np.linalg.norm(spoint)
		# 	spoint = spoint/norm
		# 	subdivpoints.SetPoint(pid, spoint[0], spoint[1], spoint[2])
		# subdivpoly.SetPoints(subdivpoints)

		# print("\nlevel", sl + 1)
		# print("Points", subdivpoly.GetNumberOfPoints())
		# print("Polys", subdivpoly.GetNumberOfPolys())
		
		# sourceObjects.append(subdivpoly)

	renderers = list()
	mappers = list()
	actors = list()
	textmappers = list()
	textactors = list()

	# Create one text property for all.
	textProperty = vtk.vtkTextProperty()
	textProperty.SetFontSize(16)
	textProperty.SetJustificationToCentered()

	backProperty = vtk.vtkProperty()
	backProperty.SetColor(colors.GetColor3d("Red"))

    # Create a source, renderer, mapper, and actor
    # for each object.
	for i in range(0, len(sourceObjects)):
		mappers.append(vtk.vtkPolyDataMapper())
		mappers[i].SetInputData(sourceObjects[i])

		actors.append(vtk.vtkActor())
		actors[i].SetMapper(mappers[i])
		actors[i].GetProperty().SetColor(colors.GetColor3d("Seashell"))
		actors[i].SetBackfaceProperty(backProperty)

		textmappers.append(vtk.vtkTextMapper())
		textmappers[i].SetInput(sourceObjects[i].GetClassName())
		textmappers[i].SetTextProperty(textProperty)

		textactors.append(vtk.vtkActor2D())
		textactors[i].SetMapper(textmappers[i])
		textactors[i].SetPosition(120, 16)
		renderers.append(vtk.vtkRenderer())

	gridDimensions = 2

    # We need a renderer even if there is no actor.
	for i in range(len(sourceObjects), gridDimensions ** 2):
		renderers.append(vtk.vtkRenderer())

	renderWindow = vtk.vtkRenderWindow()
	renderWindow.SetWindowName("Icosahedron demo")
	rendererSize = 300
	renderWindow.SetSize(rendererSize * gridDimensions, rendererSize * gridDimensions)

	for row in range(0, gridDimensions):
		for col in range(0, gridDimensions):
			index = row * gridDimensions + col
			x0 = float(col) / gridDimensions
			y0 = float(gridDimensions - row - 1) / gridDimensions
			x1 = float(col + 1) / gridDimensions
			y1 = float(gridDimensions - row) / gridDimensions
			renderWindow.AddRenderer(renderers[index])
			renderers[index].SetViewport(x0, y0, x1, y1)

			if index > (len(sourceObjects) - 1):
			    continue

			renderers[index].AddActor(actors[index])
			renderers[index].AddActor(textactors[index])
			renderers[index].SetBackground(colors.GetColor3d("BkgColor"))
			renderers[index].ResetCamera()
			renderers[index].GetActiveCamera().Azimuth(30)
			renderers[index].GetActiveCamera().Elevation(30)
			renderers[index].GetActiveCamera().Zoom(0.8)
			renderers[index].ResetCameraClippingRange()

	interactor = vtk.vtkRenderWindowInteractor()
	interactor.SetRenderWindow(renderWindow)

	renderWindow.Render()
	interactor.Start()


if __name__ == "__main__":
    main()