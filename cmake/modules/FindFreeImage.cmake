find_path(FreeImage_INCLUDE_DIRS
  NAMES FreeImage.h
)
mark_as_advanced(FreeImage_INCLUDE_DIRS)

find_library(FreeImage_LIBRARIES
  NAMES freeimage
)
mark_as_advanced(FreeImage_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FreeImage DEFAULT_MSG FreeImage_INCLUDE_DIRS FreeImage_LIBRARIES)
