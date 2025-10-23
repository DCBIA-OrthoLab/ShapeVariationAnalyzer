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

  # Use a relative path for the requirements.txt file to make it generalizable
  set(requirements_file "${CMAKE_SOURCE_DIR}/requirements.txt")
  set(pip_install_args)

  if(NOT Slicer_SOURCE_DIR)
    # Alternative python prefix for installing extension python packages
    set(python_packages_DIR "${CMAKE_BINARY_DIR}/python-packages-install")

    # Convert to native path to satisfy pip install command
    file(TO_NATIVE_PATH ${python_packages_DIR} python_packages_DIR_NATIVE_DIR)

    # Escape command argument for pip install command
    string(REGEX REPLACE "\\\\" "\\\\\\\\" python_packages_DIR_NATIVE_DIR "${python_packages_DIR_NATIVE_DIR}")

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
    INSTALL_COMMAND ${PYTHON_EXECUTABLE} -m pip install -r ${requirements_file} ${pip_install_args}
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
