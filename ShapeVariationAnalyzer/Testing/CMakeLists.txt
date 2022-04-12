set(INPUT ${CMAKE_CURRENT_SOURCE_DIR}/../Data/Input)

set(input_filenames
  g01_bent00.vtk
  g01_bent01.vtk
  g01_bent02.vtk
  g01_bent03.vtk
  g01_bent04.vtk
  g01_bent05.vtk
  g01_bent06.vtk
  g01_bent07.vtk
  g01_bent08.vtk
  g01_bent09.vtk
  test.csv
  )

foreach(filename IN LISTS input_filenames)
  # Copy file to support running tests from the build tree
  file(COPY ${INPUT}/${test_filename} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

  # Install file to support running tests from an install tree
  # TODO
endforeach()
