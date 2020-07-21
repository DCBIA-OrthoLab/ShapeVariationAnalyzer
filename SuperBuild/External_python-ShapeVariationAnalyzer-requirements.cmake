set(proj python-ShapeVariationAnalyzer-requirements)

# Set dependency list
set(${proj}_DEPENDENCIES "")
if(Slicer_SOURCE_DIR)
  set(${proj}_DEPENDENCIES
    python
    python-pip
    python-setuptools
    python-wheel
    )
  # Support Slicer older than Slicer@f348d6f from 2019-10-09
  if(EXISTS ${Slicer_SOURCE_DIR}/SuperBuild/External_python-numpy.cmake)
    list(APPEND ${proj}_DEPENDENCIES python-numpy)
  else()
    list(APPEND ${proj}_DEPENDENCIES NUMPY)
  endif()
  if(EXISTS ${Slicer_SOURCE_DIR}/SuperBuild/External_python-scipy.cmake)
    list(APPEND ${proj}_DEPENDENCIES python-scipy)
  else()
    list(APPEND ${proj}_DEPENDENCIES SciPy)
  endif()
endif()

if(NOT DEFINED Slicer_USE_SYSTEM_${proj})
  set(Slicer_USE_SYSTEM_${proj} ${Slicer_USE_SYSTEM_python})
endif()

# Include dependent projects if any
ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj DEPENDS_VAR ${proj}_DEPENDENCIES)

if(Slicer_USE_SYSTEM_${proj})
  foreach(module_name IN ITEMS sklearn)
    ExternalProject_FindPythonPackage(
      MODULE_NAME "${module_name}"
      REQUIRED
      )
  endforeach()
endif()

if(NOT Slicer_USE_SYSTEM_${proj})

  set(requirements_file ${CMAKE_BINARY_DIR}/${proj}-requirements.txt)
  file(WRITE ${requirements_file} [===[
  joblib==0.16.0 --hash=sha256:d348c5d4ae31496b2aa060d6d9b787864dd204f9480baaa52d18850cb43e9f49
  threadpoolctl==2.1.0 --hash=sha256:38b74ca20ff3bb42caca8b00055111d74159ee95c4370882bbff2b93d24da725
  # Hashes correspond to the following packages:
  # - scikit_learn-0.23.1-cp36-cp36m-win_amd64.whl
  # - scikit_learn-0.23.1-cp36-cp36m-macosx_10_9_x86_64.whl 
  # - scikit_learn-0.23.1-cp36-cp36m-manylinux1_x86_64.whl
  scikit-learn==0.23.1 --hash=sha256:e585682e37f2faa81ad6cd4472fff646bf2fd0542147bec93697a905db8e6bd2 \
                       --hash=sha256:058d213092de4384710137af1300ed0ff030b8c40459a6c6f73c31ccd274cc39 \
                       --hash=sha256:e9879ba9e64ec3add41bf201e06034162f853652ef4849b361d73b0deb3153ad
  ]===])

  set(pip_install_args)

  if(NOT Slicer_SOURCE_DIR)
    # Alternative python prefix for installing extension python packages
    set(python_packages_DIR "${CMAKE_BINARY_DIR}/python-packages-install")
    file(TO_NATIVE_PATH ${python_packages_DIR} python_packages_DIR_NATIVE_DIR)

    list(APPEND pip_install_args
      --prefix ${python_packages_DIR_NATIVE_DIR}
      )
  endif()

  ExternalProject_Add(${proj}
    ${${proj}_EP_ARGS}
    DOWNLOAD_COMMAND ""
    SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj}
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${PYTHON_EXECUTABLE} -m pip install --require-hashes -r ${requirements_file} ${pip_install_args}
    LOG_INSTALL 1
    DEPENDS
      ${${proj}_DEPENDENCIES}
    )

  ExternalProject_GenerateProjectDescription_Step(${proj}
    VERSION ${_version}
    )

  #-----------------------------------------------------------------------------
  # Launcher setting specific to build tree
  if(NOT Slicer_SOURCE_DIR)
    set(${proj}_PYTHONPATH_LAUNCHER_BUILD
      ${python_packages_DIR}/${PYTHON_STDLIB_SUBDIR}
      ${python_packages_DIR}/${PYTHON_STDLIB_SUBDIR}/lib-dynload
      ${python_packages_DIR}/${PYTHON_SITE_PACKAGES_SUBDIR}
      )
    mark_as_superbuild(
      VARS ${proj}_PYTHONPATH_LAUNCHER_BUILD
      LABELS "PYTHONPATH_LAUNCHER_BUILD"
      )

    mark_as_superbuild(python_packages_DIR:PATH)
  endif()

else()
  ExternalProject_Add_Empty(${proj} DEPENDS ${${proj}_DEPENDENCIES})
endif()
