find_path(cnpy_INCLUDE_DIRS
  NAMES cnpy.h
  PATH_SUFFIXES include
)
mark_as_advanced(cnpy_INCLUDE_DIRS)

find_library(cnpy_LIBRARIES
  NAMES cnpy
  PATH_SUFFIXES lib lib64
)
mark_as_advanced(cnpy_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cnpy DEFAULT_MSG cnpy_INCLUDE_DIRS cnpy_LIBRARIES)
