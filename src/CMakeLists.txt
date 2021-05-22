find_package(Boost REQUIRED COMPONENTS filesystem)

add_library(base INTERFACE)
target_include_directories(base
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(base INTERFACE cxx_std_11)
target_link_libraries(base INTERFACE Boost::filesystem Metal jtuple)
target_compile_definitions(base INTERFACE
  COPY_BY_ASSIGN_BELOW_NUMBER_OF_BYTES=${COPY_BY_ASSIGN_BELOW_NUMBER_OF_BYTES}
  BOOST_PP_VARIADICS=1 # Force this option for versions of boost that disable it for nvcc
)
install(
  TARGETS base
  EXPORT install_targets
)



set(OpenGL_GL_PREFERENCE GLVND)



if(CMAKE_CUDA_COMPILER)
  find_package(CUDAToolkit REQUIRED)

  add_library(interface.cublas Cublas.cu)
  target_link_libraries(interface.cublas PUBLIC CUDA::cublas base)
  target_compile_definitions(interface.cublas PUBLIC CUBLAS_INCLUDED)
  install(
    TARGETS interface.cublas
    EXPORT install_targets
    LIBRARY DESTINATION ${INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${INSTALL_LIBDIR}
  )

  find_package(CUDNN QUIET)
  if(CUDNN_FOUND)
    add_library(interface.cudnn Cudnn.cu)
    target_link_libraries(interface.cudnn PUBLIC ${CUDNN_LIBRARIES} base)
    target_include_directories(interface.cudnn PUBLIC ${CUDNN_INCLUDE_DIRS})
    target_compile_definitions(interface.cudnn PUBLIC CUDNN_INCLUDED)
    install(
      TARGETS interface.cudnn
      EXPORT install_targets
      LIBRARY DESTINATION ${INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${INSTALL_LIBDIR}
    )
  endif()
endif()


find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  add_library(interface.openmp INTERFACE)
  target_link_libraries(interface.openmp INTERFACE OpenMP::OpenMP_CXX base)
  target_compile_definitions(interface.openmp INTERFACE OPENMP_INCLUDED)
  target_compile_options(interface.openmp INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>) # This isn't enabled for cuda files by default
  install(
    TARGETS interface.openmp
    EXPORT install_targets
  )
endif()


find_package(TBB QUIET)
if(TBB_FOUND)
  add_library(interface.tbb INTERFACE)
  target_link_libraries(interface.tbb INTERFACE TBB::tbb base)
  target_compile_definitions(interface.tbb INTERFACE TBB_INCLUDED)
  install(
    TARGETS interface.tbb
    EXPORT install_targets
  )
endif()


find_package(OpenCV QUIET)
if(OpenCV_FOUND)
  add_library(interface.opencv INTERFACE)
  target_link_libraries(interface.opencv INTERFACE ${OpenCV_LIBS} base)
  target_include_directories(interface.opencv INTERFACE ${OpenCV_INCLUDE_DIRS})
  target_compile_definitions(interface.opencv INTERFACE OPENCV_INCLUDED)
  install(
    TARGETS interface.opencv
    EXPORT install_targets
  )
endif()


find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
  add_library(interface.eigen INTERFACE)
  target_link_libraries(interface.eigen INTERFACE Eigen3::Eigen base)
  target_compile_definitions(interface.eigen INTERFACE EIGEN_INCLUDED)
  target_compile_options(interface.eigen INTERFACE
    #  Disable warning: __device__ annotation is ignored on a function("...") that is explicitly defaulted on its first declaration
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag_suppress=esa_on_defaulted_function_ignored>
  )
  install(
    TARGETS interface.eigen
    EXPORT install_targets
  )
endif()


find_package(PCL COMPONENTS io common kdtree QUIET) # keypoints filters range_image registration sample_consensus segmentation features surface octree visualization
if(PCL_FOUND)
  add_library(interface.pcl INTERFACE)
  target_link_libraries(interface.pcl INTERFACE PCL::all base interface.eigen)
  target_compile_definitions(interface.pcl INTERFACE PCL_INCLUDED)
  install(TARGETS interface.pcl
    EXPORT install_targets
  )
endif()


find_package(tf2 QUIET)
if(tf2_FOUND)
  add_library(interface.tf2 INTERFACE)
  target_link_libraries(interface.tf2 INTERFACE base ${tf2_LIBS})
  target_include_directories(interface.tf2 INTERFACE ${tf2_INCLUDE_DIRS})
  target_compile_definitions(interface.tf2 INTERFACE TF2_INCLUDED)
  install(
    TARGETS interface.tf2
    EXPORT install_targets
  )
endif()


find_package(tf QUIET)
if(tf_FOUND)
  add_library(interface.tf INTERFACE)
  target_link_libraries(interface.tf INTERFACE base ${tf_LIBS})
  target_include_directories(interface.tf INTERFACE ${tf_INCLUDE_DIRS})
  target_compile_definitions(interface.tf INTERFACE TF_INCLUDED)
  install(
    TARGETS interface.tf
    EXPORT install_targets
  )
endif()


find_package(FreeImage QUIET)
if(FreeImage_FOUND)
  add_library(interface.freeimage INTERFACE)
  target_link_libraries(interface.freeimage INTERFACE base ${FreeImage_LIBRARIES})
  target_include_directories(interface.freeimage INTERFACE ${FreeImage_INCLUDE_DIRS})
  target_compile_definitions(interface.freeimage INTERFACE FREEIMAGE_INCLUDED)
  install(
    TARGETS interface.freeimage
    EXPORT install_targets
  )
endif()


find_package(sensor_msgs QUIET)
if(sensor_msgs_FOUND)
  add_library(interface.sensor_msgs INTERFACE)
  target_link_libraries(interface.sensor_msgs INTERFACE ${sensor_msgs_LIBS} base)
  target_include_directories(interface.sensor_msgs INTERFACE ${sensor_msgs_INCLUDE_DIRS})
  target_compile_definitions(interface.sensor_msgs INTERFACE SENSOR_MSGS_INCLUDED)
  install(
    TARGETS interface.sensor_msgs
    EXPORT install_targets
  )
endif()


find_package(orocos_kdl QUIET)
if(orocos_kdl_FOUND)
  # Remove hardcoded include directory for eigen in orocos_kdl-config.cmake, add interface.eigen instead
  list(FILTER orocos_kdl_INCLUDE_DIRS EXCLUDE REGEX "eigen3$")
  add_library(interface.orocos_kdl INTERFACE)
  target_link_libraries(interface.orocos_kdl INTERFACE interface.eigen ${orocos_kdl_LIBRARIES} base)
  target_include_directories(interface.orocos_kdl INTERFACE ${orocos_kdl_INCLUDE_DIRS})
  target_compile_definitions(interface.orocos_kdl INTERFACE OROCOS_KDL_INCLUDED)
  install(
    TARGETS interface.orocos_kdl
    EXPORT install_targets
  )
endif()


find_package(tinyply QUIET)
if(tinyply_FOUND)
  add_library(interface.tinyply INTERFACE)
  target_link_libraries(interface.tinyply INTERFACE tinyply base)
  target_compile_definitions(interface.tinyply INTERFACE TINYPLY_INCLUDED)
  install(
    TARGETS interface.tinyply
    EXPORT install_targets
  )
endif()


find_package(cnpy QUIET)
if(cnpy_FOUND)
  add_library(interface.cnpy INTERFACE)
  target_link_libraries(interface.cnpy INTERFACE ${cnpy_LIBRARIES} z base)
  target_include_directories(interface.cnpy INTERFACE ${cnpy_INCLUDE_DIRS})
  target_compile_definitions(interface.cnpy INTERFACE CNPY_INCLUDED)
  install(
    TARGETS interface.cnpy
    EXPORT install_targets
  )
endif()


set(Python3_FIND_STRATEGY LOCATION)
find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(Python3_Interpreter_FOUND AND Python3_Development_FOUND)
  add_library(interface.python INTERFACE)
  target_link_libraries(interface.python INTERFACE Python3::Python Python3::Module base)
  target_compile_definitions(interface.python INTERFACE PYTHON_INCLUDED)
  install(
    TARGETS interface.python
    EXPORT install_targets
  )
endif()


if(TARGET interface.python)
  set(DEP_NAME)
  find_package(Boost COMPONENTS python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR} QUIET)
  if(Boost_python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}_FOUND)
    # Boost >= 1.67
    set(DEP_NAME python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR})
  else()
    # Boost < 1.67
    find_package(Boost COMPONENTS python QUIET)
    if(Boost_python_FOUND)
      set(DEP_NAME python)
    endif()
  endif()
  if(DEP_NAME)
    add_library(interface.boost.python INTERFACE)
    target_link_libraries(interface.boost.python INTERFACE interface.python base Boost::${DEP_NAME})
    target_compile_definitions(interface.boost.python INTERFACE
      BOOST_PYTHON_INCLUDED
      BOOST_BIND_GLOBAL_PLACEHOLDERS # Avoid deprecation warning for code in boost/python.hpp
    )
    install(
      TARGETS interface.boost.python
      EXPORT install_targets
    )
  endif()
endif()


if(TARGET interface.boost.python)
  set(DEP_NAME)
  find_package(Boost COMPONENTS numpy${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR} QUIET)
  if(Boost_numpy${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}_FOUND)
    # Boost >= 1.67
    set(DEP_NAME numpy${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR})
  else()
    # Boost < 1.67
    find_package(Boost COMPONENTS numpy QUIET)
    if(Boost_numpy_FOUND)
      set(DEP_NAME numpy)
    endif()
  endif()
  if(DEP_NAME)
    add_library(interface.boost.numpy INTERFACE)
    target_link_libraries(interface.boost.numpy INTERFACE interface.boost.python Boost::${DEP_NAME} base)
    target_compile_definitions(interface.boost.numpy INTERFACE BOOST_NUMPY_INCLUDED)
    install(
      TARGETS interface.boost.numpy
      EXPORT install_targets
    )
  endif()
endif()


find_package(SOIL QUIET)
find_package(OpenGL QUIET)
if(SOIL_FOUND AND OpenGL_FOUND)
  add_library(interface.soil INTERFACE)
  target_link_libraries(interface.soil INTERFACE ${SOIL_LIBRARIES} OpenGL::GL base)
  target_include_directories(interface.soil INTERFACE ${SOIL_INCLUDE_DIRS})
  target_compile_definitions(interface.soil INTERFACE SOIL_INCLUDED)
  install(
    TARGETS interface.soil
    EXPORT install_targets
  )
endif()


find_package(dlpack QUIET)
if(dlpack_FOUND)
  add_library(interface.dlpack INTERFACE)
  target_link_libraries(interface.dlpack INTERFACE base dlpack::dlpack)
  target_compile_definitions(interface.dlpack INTERFACE DLPACK_INCLUDED)
  install(
    TARGETS interface.dlpack
    EXPORT install_targets
  )
endif()


find_package(NVOF QUIET)
if(NVOF_FOUND)
  add_library(interface.nvof INTERFACE)
  target_link_libraries(interface.nvof INTERFACE base NVOF::nvof)
  target_compile_definitions(interface.nvof INTERFACE NVOF_INCLUDED)
  install(
    TARGETS interface.nvof
    EXPORT install_targets
  )
endif()


find_package(cereal QUIET)
if(cereal_FOUND)
  add_library(interface.cereal INTERFACE)
  target_link_libraries(interface.cereal INTERFACE base cereal::cereal)
  target_compile_definitions(interface.cereal INTERFACE CEREAL_INCLUDED)
  install(
    TARGETS interface.cereal
    EXPORT install_targets
  )
endif()


find_package(cppflow QUIET)
if(cppflow_FOUND AND TARGET interface.dlpack)
  add_library(interface.cppflow INTERFACE)
  target_link_libraries(interface.cppflow INTERFACE base cppflow::cppflow interface.dlpack)
  target_compile_definitions(interface.cppflow INTERFACE CPPFLOW_INCLUDED)
  install(
    TARGETS interface.cppflow
    EXPORT install_targets
  )
endif()


find_package(g2o QUIET)
if(g2o_FOUND AND TARGET interface.eigen)
  add_library(interface.g2o INTERFACE)
  target_link_libraries(interface.g2o INTERFACE base g2o::core g2o::types_sba interface.eigen)
  target_compile_definitions(interface.g2o INTERFACE G2O_INCLUDED)
  install(
    TARGETS interface.g2o
    EXPORT install_targets
  )
endif()


message(STATUS "Available interfaces:")
if(TARGET interface.cudnn)
  message(STATUS "  Cudnn")
endif()
if(TARGET interface.cublas)
  message(STATUS "  Cublas")
endif()
if(TARGET interface.openmp)
  message(STATUS "  OpenMP")
endif()
if(TARGET interface.tbb)
  message(STATUS "  TBB")
endif()
if(TARGET interface.opencv)
  message(STATUS "  OpenCV")
endif()
if(TARGET interface.pcl)
  message(STATUS "  PCL")
endif()
if(TARGET interface.eigen)
  message(STATUS "  Eigen")
endif()
if(TARGET interface.tf2)
  message(STATUS "  tf2")
endif()
if(TARGET interface.tf)
  message(STATUS "  tf")
endif()
if(TARGET interface.freeimage)
  message(STATUS "  FreeImage")
endif()
if(TARGET interface.sensor_msgs)
  message(STATUS "  sensor_msgs")
endif()
if(TARGET interface.orocos_kdl)
  message(STATUS "  orocos_kdl")
endif()
if(TARGET interface.tinyply)
  message(STATUS "  tinyply")
endif()
if(TARGET interface.cnpy)
  message(STATUS "  cnpy")
endif()
if(TARGET interface.boost.python)
  message(STATUS "  boost.python")
endif()
if(TARGET interface.boost.numpy)
  message(STATUS "  boost.numpy")
endif()
if(TARGET interface.soil)
  message(STATUS "  SOIL")
endif()
if(TARGET interface.dlpack)
  message(STATUS "  DLPack")
endif()
if(TARGET interface.nvof)
  message(STATUS "  NVOF")
endif()
if(TARGET interface.cereal)
  message(STATUS "  Cereal")
endif()
if(TARGET interface.cppflow)
  message(STATUS "  CppFlow")
endif()
if(TARGET interface.g2o)
  message(STATUS "  g2o")
endif()
