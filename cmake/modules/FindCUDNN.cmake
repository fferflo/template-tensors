find_path(CUDNN_INCLUDE_DIR
  NAMES cudnn.h
  HINTS ${CMAKE_CUDA_COMPILER}/../..
  PATH_SUFFIXES include
)
mark_as_advanced(CUDNN_INCLUDE_DIR)

find_library(CUDNN_LIBRARY cudnn
  HINTS ${CMAKE_CUDA_COMPILER}/../..
  PATH_SUFFIXES lib lib64
)
mark_as_advanced(CUDNN_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_INCLUDE_DIR CUDNN_LIBRARY)

set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
