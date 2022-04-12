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
  # [joblib]
  joblib==1.1.0 --hash=sha256:f21f109b3c7ff9d95f8387f752d0d9c34a02aa2f7060c2135f465da0e5160ff6
  # [/joblib]
  # [threadpoolctl]
  threadpoolctl==3.1.0 --hash=sha256:8b99adda265feb6773280df41eece7b2e6561b772d21ffd52e372f999024907b
  # [/threadpoolctl]
  # [scikit-learn]
  # Hashes correspond to the following packages:
  #  - scikit_learn-1.0.2-cp39-cp39-macosx_10_13_x86_64.whl
  #  - scikit_learn-1.0.2-cp39-cp39-macosx_12_0_arm64.whl
  #  - scikit_learn-1.0.2-cp39-cp39-manylinux_2_12_x86_64.manylinux2010_x86_64.whl
  #  - scikit_learn-1.0.2-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
  #  - scikit_learn-1.0.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  #  - scikit_learn-1.0.2-cp39-cp39-win_amd64.whl
  scikit-learn==1.0.2 --hash=sha256:a90b60048f9ffdd962d2ad2fb16367a87ac34d76e02550968719eb7b5716fd10 \
                      --hash=sha256:7a93c1292799620df90348800d5ac06f3794c1316ca247525fa31169f6d25855 \
                      --hash=sha256:55f2f3a8414e14fbee03782f9fe16cca0f141d639d2b1c1a36779fa069e1db57 \
                      --hash=sha256:80095a1e4b93bd33261ef03b9bc86d6db649f988ea4dbcf7110d0cded8d7213d \
                      --hash=sha256:ff746a69ff2ef25f62b36338c615dd15954ddc3ab8e73530237dd73235e76d62 \
                      --hash=sha256:b54a62c6e318ddbfa7d22c383466d38d2ee770ebdb5ddb668d56a099f6eaf75f
  # [/scikit-learn]
  ]===])

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
