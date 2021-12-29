find_package(Boost REQUIRED
  unit_test_framework
)

add_custom_target(tests)

macro(addIndividualTest TARGET TEST_NAME)
  add_executable(${TARGET} EXCLUDE_FROM_ALL ${TEST_NAME})
  target_link_libraries(${TARGET} Boost::unit_test_framework base)
  add_test(NAME ${TARGET} WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin/test" COMMAND "${CMAKE_BINARY_DIR}/bin/test/${TARGET}")
  set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/test")
  target_compile_definitions(${TARGET} PRIVATE BOOST_TEST_MODULE="${TARGET}")
  target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_dependencies(tests ${TARGET})
endmacro()

macro(addTestForcable TARGET TEST_NAME CUDA_ONLY)
  configure_file(${TEST_NAME}.h ${TEST_NAME}.cpp)
  configure_file(${TEST_NAME}.h ${TEST_NAME}.cu)

  set(TARGETS "")

  if(NOT ${CUDA_ONLY})
    addIndividualTest(${TARGET}_nocuda ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.cpp)
    list(APPEND TARGETS ${TARGET}_nocuda)
  endif()

  if(CMAKE_CUDA_COMPILER)
    addIndividualTest(${TARGET}_cuda ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.cu)
    if(NOT ${CUDA_ONLY})
      # Compile non-cuda tensor tests before cuda tensor tests to get better gcc compiler output
      add_dependencies(${TARGET}_cuda ${TARGET}_nocuda)
    endif()
    list(APPEND TARGETS ${TARGET}_cuda)
  endif()
endmacro()

macro(addCudaTest TARGET TEST_NAME_ARG)
  addTestForcable(${TARGET} ${TEST_NAME_ARG} TRUE)
endmacro()

macro(addTest TARGET TEST_NAME_ARG)
  addTestForcable(${TARGET} ${TEST_NAME_ARG} FALSE)
endmacro()

find_package(Python3 COMPONENTS Interpreter Development)

macro(addIndividualPythonTest TARGET TEST_LIB_NAME TEST_PY_NAME)
  Python3_add_library(${TARGET} EXCLUDE_FROM_ALL MODULE ${TEST_LIB_NAME})
  target_link_libraries(${TARGET} PRIVATE base interface.python)
  set(TEST_MODULE ${TARGET}) # Used in configure_file
  configure_file(${TEST_PY_NAME} "${CMAKE_BINARY_DIR}/bin/test/${TARGET}.py")
  add_test(NAME ${TARGET} WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin/test" COMMAND python3 "${CMAKE_BINARY_DIR}/bin/test/${TARGET}.py")
  set_target_properties(${TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/test")
  target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_dependencies(tests ${TARGET})
endmacro()

macro(addPythonTestForcable TARGET TEST_NAME_ARG CUDA_ONLY)
  set(TEST_LIB_NAME ${TEST_NAME_ARG}Lib)
  set(TEST_PY_NAME ${TEST_NAME_ARG}.py)

  set(TARGETS "")

  if(NOT ${CUDA_ONLY})
    set(TEST_MODULE ${TARGET}_nocuda_lib) # Used in configure_file
    configure_file(${TEST_LIB_NAME}.h ${TEST_LIB_NAME}_nocuda.cpp)
    addIndividualPythonTest(${TARGET}_nocuda_lib ${CMAKE_CURRENT_BINARY_DIR}/${TEST_LIB_NAME}_nocuda.cpp ${TEST_PY_NAME})
    list(APPEND TARGETS ${TARGET}_nocuda_lib)
  endif()

  if(CMAKE_CUDA_COMPILER)
    set(TEST_MODULE ${TARGET}_cuda_lib) # Used in configure_file
    configure_file(${TEST_LIB_NAME}.h ${TEST_LIB_NAME}_cuda.cu)
    addIndividualPythonTest(${TARGET}_cuda_lib ${CMAKE_CURRENT_BINARY_DIR}/${TEST_LIB_NAME}_cuda.cu ${TEST_PY_NAME})
    if(NOT ${CUDA_ONLY})
      # Compile non-cuda tensor tests before cuda tensor tests to get better gcc compiler output
      add_dependencies(${TARGET}_cuda_lib ${TARGET}_nocuda_lib)
    endif()
    list(APPEND TARGETS ${TARGET}_cuda_lib)
  endif()
endmacro()

macro(addCudaPythonTest TARGET TEST_NAME_ARG)
  addPythonTestForcable(${TARGET} ${TEST_NAME_ARG} TRUE)
endmacro()

macro(addPythonTest TARGET TEST_NAME_ARG)
  addPythonTestForcable(${TARGET} ${TEST_NAME_ARG} FALSE)
endmacro()

set(TEST_DIR "${CMAKE_BINARY_DIR}/test_dir")
file(MAKE_DIRECTORY ${TEST_DIR})



######################## ACQUIRE TEST DATA ########################

set(AIRPLANE_TIFF "${CMAKE_BINARY_DIR}/data/airplane.tiff")
set(AIRPLANE_TIFF_WIDTH 512)
set(AIRPLANE_TIFF_HEIGHT 512)
ExternalProject_Add(
    airplane_tiff
    PREFIX "tmp"
    DOWNLOAD_NAME "airplane.tiff"
    DOWNLOAD_NO_EXTRACT ON
    DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/data"
    URL http://sipi.usc.edu/database/download.php?vol=misc&img=4.2.05
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL 1
)

set(AIRPLANE_PNG "${CMAKE_BINARY_DIR}/data/airplane.png")
set(AIRPLANE_PNG_WIDTH 512)
set(AIRPLANE_PNG_HEIGHT 512)
add_custom_target(airplane_png
  COMMAND convert ${AIRPLANE_TIFF} ${AIRPLANE_PNG}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  VERBATIM
)
add_dependencies(airplane_png airplane_tiff)

set(AIRPLANE_HALF_PNG "${CMAKE_BINARY_DIR}/data/airplane_half.png")
set(AIRPLANE_HALF_PNG_WIDTH 512)
set(AIRPLANE_HALF_PNG_HEIGHT 256)
add_custom_target(airplane_half_png
  COMMAND convert ${AIRPLANE_PNG} -crop 512x256+0+0 ${AIRPLANE_HALF_PNG}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  VERBATIM
)
add_dependencies(airplane_half_png airplane_png)

set(PCD_TABLE_SCENE_LMS400 "${CMAKE_BINARY_DIR}/data/table_scene_lms400.pcd")
ExternalProject_Add(
    table_scene_lms400
    PREFIX "tmp"
    DOWNLOAD_NAME "table_scene_lms400.pcd"
    DOWNLOAD_NO_EXTRACT ON
    DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/data"
    URL https://github.com/PointCloudLibrary/data/blob/master/tutorials/table_scene_lms400.pcd?raw=true
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL 1
)
set(PCD_TABLE_SCENE_LMS400_SUBSAMPLED "${CMAKE_BINARY_DIR}/data/table_scene_lms400_subsampled.pcd")
add_custom_target(table_scene_lms400_subsampled
  COMMAND if test ! -f ${PCD_TABLE_SCENE_LMS400_SUBSAMPLED}\; then pcl_voxel_grid ${PCD_TABLE_SCENE_LMS400} ${PCD_TABLE_SCENE_LMS400_SUBSAMPLED} -leaf 0.01,0.01,0.01\; fi\;
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)
add_dependencies(table_scene_lms400_subsampled table_scene_lms400)
set(PCD_TABLE_SCENE_LMS400_SUBSAMPLED_NORMALS "${CMAKE_BINARY_DIR}/data/table_scene_lms400_subsampled_normals.pcd")
add_custom_target(table_scene_lms400_subsampled_normals
  COMMAND if test ! -f ${PCD_TABLE_SCENE_LMS400_SUBSAMPLED_NORMALS}\; then pcl_normal_estimation ${PCD_TABLE_SCENE_LMS400_SUBSAMPLED} ${PCD_TABLE_SCENE_LMS400_SUBSAMPLED_NORMALS} -radius 0.05\; fi\;
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)
add_dependencies(table_scene_lms400_subsampled_normals table_scene_lms400_subsampled)

set(CHAIR_BINVOX "${CMAKE_BINARY_DIR}/data/chair.binvox")
ExternalProject_Add(
    chair_binvox
    PREFIX "tmp"
    DOWNLOAD_NAME "chair.binvox"
    DOWNLOAD_NO_EXTRACT ON
    DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/data"
    URL https://github.com/dimatura/binvox-rw-py/blob/67844ea547963717189269e49edbca9dabb43e4a/chair.binvox?raw=true
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL 1
)

set(MIDDLEBURY_FLOW_PATH "${CMAKE_BINARY_DIR}/data/middlebury_flow")
ExternalProject_Add(
    middlebury_flow_gt
    PREFIX "tmp"
    SOURCE_DIR "${MIDDLEBURY_FLOW_PATH}/gt"
    URL https://vision.middlebury.edu/flow/data/comp/zip/other-gt-flow.zip
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL 1
)
ExternalProject_Add(
    middlebury_flow_color
    PREFIX "tmp"
    SOURCE_DIR "${MIDDLEBURY_FLOW_PATH}/color"
    URL https://vision.middlebury.edu/flow/data/comp/zip/other-color-twoframes.zip
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL 1
)
add_custom_target(middlebury_flow DEPENDS middlebury_flow_gt middlebury_flow_color)



######################## TEST BASICS ########################

addTest(test_aggregator util/TestAggregator)
addTest(test_numeric numeric/TestNumeric)
addTest(test_field field/TestField)
addTest(test_iterator util/TestIterator)
addTest(test_ptr util/TestPtr)
addTest(test_dispatch util/TestDispatch)

addTest(test_binvox file/TestBinvox)
foreach(TARGET ${TARGETS})
  add_dependencies(${TARGET} chair_binvox)
  target_compile_definitions(${TARGET} PRIVATE BINVOX_FILE="${CHAIR_BINVOX}")
endforeach()

addTest(test_flo file/TestFlo)
foreach(TARGET ${TARGETS})
  add_dependencies(${TARGET} middlebury_flow)
  target_compile_definitions(${TARGET} PRIVATE MIDDLEBURY_FLOW_PATH="${MIDDLEBURY_FLOW_PATH}")
  if(TARGET interface.opencv)
    target_link_libraries(${TARGET} interface.opencv)
  endif()
endforeach()

addTest(test_random tensor/TestRandom)
addTest(test_sparse tensor/TestSparse)
addTest(test_reduction_broadcasting tensor/TestReductionBroadcasting)
addTest(test_gaussian_method tensor/TestGaussianMethod)
addTest(test_index_strategy tensor/TestIndexStrategy)
addTest(test_elwise_ops tensor/TestElwiseOps)
addTest(test_misc_ops tensor/TestMiscOps)
addTest(test_masking tensor/TestMasking)
addTest(test_matrix_ops tensor/TestMatrixOps)
addTest(test_shape_ops tensor/TestShapeOps)
addTest(test_basics tensor/TestBasics)
addTest(test_for_each tensor/TestForEach)
addTest(test_convolution tensor/TestConvolution)
addCudaTest(test_assignment tensor/TestAssignment)

addTest(test_transform geometry/TestTransform)
addTest(test_projections geometry/TestProjections)
addTest(test_region_growing geometry/TestRegionGrowing)

addTest(test_std interface/TestStd)

addTest(test_sort_grid point_cloud/TestSortGrid)

if(CMAKE_CUDA_COMPILER)
  addCudaTest(test_cuda cuda/TestCuda)
  addCudaTest(test_thrust interface/TestThrust)
  addCudaTest(test_device_for_each tensor/TestDeviceForEach)
  addCudaTest(test_host_device_transfer tensor/TestHostDeviceTransfer)
  addCudaTest(test_render geometry/TestRender)
endif()



######################## TEST LINKING ########################

add_executable(test_linking_nocuda EXCLUDE_FROM_ALL linking/TestLink1.cpp linking/TestLink2.cpp)
target_link_libraries(test_linking_nocuda base)
set_target_properties(test_linking_nocuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/test")
target_include_directories(test_linking_nocuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(tests test_linking_nocuda)
if(CMAKE_CUDA_COMPILER)
  add_executable(test_linking_cuda EXCLUDE_FROM_ALL linking/TestLink1.cu linking/TestLink2.cu)
  target_link_libraries(test_linking_cuda base)
  set_target_properties(test_linking_cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/test")
  target_include_directories(test_linking_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_dependencies(tests test_linking_cuda)
endif()



######################## TEST INTERFACES ########################

addTest(test_jtuple interface/TestJtuple)

if(TARGET interface.cublas)
  addCudaTest(test_cublas interface/TestCublas)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.cublas)
  endforeach()
endif()

if(TARGET interface.cudnn)
  addCudaTest(test_cudnn interface/TestCudnn)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.cudnn)
  endforeach()
endif()

if(TARGET interface.opencv)
  addTest(test_opencv interface/TestOpenCv)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.opencv)
    add_dependencies(${TARGET} airplane_half_png)
    target_compile_definitions(${TARGET} PRIVATE
      IMAGE_PATH="${AIRPLANE_HALF_PNG}"
      IMAGE_WIDTH=${AIRPLANE_HALF_PNG_WIDTH}
      IMAGE_HEIGHT=${AIRPLANE_HALF_PNG_HEIGHT}
    )
  endforeach()
endif()

if(TARGET interface.openmp)
  addTest(test_openmp interface/TestOpenMp)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.openmp)
  endforeach()
endif()

if(TARGET interface.tbb)
  addTest(test_tbb interface/TestTbb)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.tbb)
  endforeach()
endif()

if(TARGET interface.pcl)
  addTest(test_pcl interface/TestPcl)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.pcl)
    add_dependencies(${TARGET} table_scene_lms400_subsampled)
    target_compile_definitions(${TARGET} PRIVATE POINT_CLOUD_FILE="${PCD_TABLE_SCENE_LMS400_SUBSAMPLED}")
  endforeach()
endif()

if(TARGET interface.eigen)
  addTest(test_eigen interface/TestEigen)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.eigen)
  endforeach()
endif()

if(TARGET interface.tf2)
  addTest(test_tf2 interface/TestTf2)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.tf2)
  endforeach()
endif()

if(TARGET interface.tf)
  addTest(test_tf interface/TestTf)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.tf)
  endforeach()
endif()

if(TARGET interface.freeimage)
  addTest(test_freeimage interface/TestFreeImage)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.freeimage)
    if(TARGET interface.opencv)
      target_link_libraries(${TARGET} interface.opencv)
    endif()
    add_dependencies(${TARGET} airplane_half_png)
    target_compile_definitions(${TARGET} PRIVATE
      IMAGE_PATH="${AIRPLANE_HALF_PNG}"
      IMAGE_WIDTH=${AIRPLANE_HALF_PNG_WIDTH}
      IMAGE_HEIGHT=${AIRPLANE_HALF_PNG_HEIGHT}
    )
  endforeach()
endif()

if(TARGET interface.sensor_msgs)
  addTest(test_sensor_msgs interface/TestSensorMsgs)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.sensor_msgs)
  endforeach()
endif()

if(TARGET interface.orocos_kdl)
  addTest(test_orocos_kdl interface/TestOrocosKdl)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.orocos_kdl)
  endforeach()
endif()

if(TARGET interface.tinyply)
  addTest(test_tinyply interface/TestTinyply)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.tinyply)
    target_compile_definitions(${TARGET} PRIVATE TEST_DIR="${TEST_DIR}")
  endforeach()
endif()

if(TARGET interface.cnpy)
  addTest(test_cnpy interface/TestCnpy)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.cnpy)
    target_compile_definitions(${TARGET} PRIVATE TEST_DIR="${TEST_DIR}")
  endforeach()
endif()

if(TARGET interface.python.boost_numpy)
  addTest(test_boost_numpy interface/boost/numpy/TestBoostNumpy)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.python.boost_numpy)
  endforeach()

  addPythonTest(test_boost_numpy_py interface/boost/numpy/TestBoostNumpy)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} PRIVATE interface.python.boost_numpy)
  endforeach()
endif()

if(TARGET interface.python.boost)
  addCudaPythonTest(test_boost_cupy_py interface/boost/cupy/TestBoostCupy)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} PRIVATE interface.python.boost)
  endforeach()
endif()

if(TARGET interface.dlpack)
  addTest(test_dlpack interface/TestDlPack)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.dlpack)
  endforeach()

  if(TARGET interface.python.boost)
    addPythonTest(test_boost_dlpack_py interface/boost/dlpack/TestBoostDlPack)
    foreach(TARGET ${TARGETS})
      target_link_libraries(${TARGET} PRIVATE interface.dlpack interface.python.boost)
    endforeach()
  endif()
endif()

if(TARGET interface.python.boost AND TARGET interface.dlpack)
  addPythonTest(test_boost_tensorflow_py interface/boost/tensorflow/TestBoostTensorflow)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} PRIVATE interface.python.boost interface.dlpack)
  endforeach()
endif()

if(TARGET interface.soil)
  addTest(test_soil interface/TestSoil)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.soil)
    if(TARGET interface.opencv)
      target_link_libraries(${TARGET} interface.opencv)
    endif()
    add_dependencies(${TARGET} airplane_half_png)
    target_compile_definitions(${TARGET} PRIVATE
      IMAGE_PATH="${AIRPLANE_HALF_PNG}"
      IMAGE_WIDTH=${AIRPLANE_HALF_PNG_WIDTH}
      IMAGE_HEIGHT=${AIRPLANE_HALF_PNG_HEIGHT}
    )
  endforeach()
endif()

if(TARGET interface.nvof)
  addCudaTest(test_nvof interface/TestNVOF)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.nvof)
    if(TARGET interface.opencv)
      target_link_libraries(${TARGET} interface.opencv)
    endif()
    add_dependencies(${TARGET} airplane_png)
    target_compile_definitions(${TARGET} PRIVATE IMAGE_PATH="${AIRPLANE_PNG}")
  endforeach()
endif()

if(TARGET interface.cereal)
  addTest(test_cereal interface/TestCereal)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.cereal)
    target_compile_definitions(${TARGET} PRIVATE TEST_DIR="${TEST_DIR}")
  endforeach()
endif()

if(TARGET interface.cppflow)
  addTest(test_cppflow interface/TestCppFlow)
  foreach(TARGET ${TARGETS})
    target_link_libraries(${TARGET} interface.cppflow)
  endforeach()
endif()
