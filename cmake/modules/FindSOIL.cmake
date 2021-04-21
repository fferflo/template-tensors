find_path(SOIL_INCLUDE_DIR
  NAMES SOIL/SOIL.h
)
mark_as_advanced(SOIL_INCLUDE_DIR)

find_library(SOIL_LIBRARY
  NAMES SOIL
)
mark_as_advanced(SOIL_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SOIL DEFAULT_MSG SOIL_INCLUDE_DIR SOIL_LIBRARY)

set(SOIL_LIBRARIES ${SOIL_LIBRARY})
set(SOIL_INCLUDE_DIRS ${SOIL_INCLUDE_DIR})
