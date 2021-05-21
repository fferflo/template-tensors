cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(template-tensors LANGUAGES CXX)

# Options
set(COPY_BY_ASSIGN_BELOW_NUMBER_OF_BYTES 512 CACHE STRING "Copy memory in mem::copy by assign-loop rather than memcpy (or device-specific alternatives) when less than this number of bytes is copied")
set(INSTALL_HEADERS_AS_SYMLINK OFF CACHE BOOL "Installs headers as a symlink to the source directory instead of copying all files to install directory")



if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-backtrace-limit=0 -Wall -Wextra -Wno-unused-parameter -march=native")
endif()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W4 -wd4100 -wd4127") # Disable warnings: unused parameter, conditional expression is constant
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

# Enable CUDA as a language only if it can be found
find_program(NVCC_PATH nvcc)
if(NOT ${NVCC_PATH} STREQUAL "NVCC_PATH-NOTFOUND")
  enable_language(CUDA)

  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
  endif()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ftemplate-backtrace-limit=0 --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -DDEBUG -g -G")

  set(CMAKE_CUDA_ARCHITECTURES OFF)
endif()

enable_testing()

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake/modules")

include(GNUInstallDirs)
include(ExternalProject)

set(PACKAGE_NAME TemplateTensors)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/${PACKAGE_NAME}/cmake)
set(INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR}/${PACKAGE_NAME})



include(extern/dependencies.cmake)
add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(test)



# Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/doc/doxyfile @ONLY)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doc/doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc
    COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)



# Install headers
if(INSTALL_HEADERS_AS_SYMLINK)
  install(CODE "file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})")
  install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/include/template_tensors ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/template_tensors)")
  install(CODE "message(\"-- Installed symlink: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/template_tensors -> ${CMAKE_CURRENT_SOURCE_DIR}/include/template_tensors\")")
else()
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()



# Install targets file
install(EXPORT install_targets
  FILE
    ${PACKAGE_NAME}Targets.cmake
  NAMESPACE
    ${PACKAGE_NAME}::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)
# Install ${PACKAGE_NAME}Config.cmake
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PACKAGE_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)
# Install find modules
install(DIRECTORY cmake/modules/ DESTINATION ${INSTALL_CONFIGDIR}/modules)
