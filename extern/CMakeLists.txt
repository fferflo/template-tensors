function(git_submodule NAME)
  find_package(Git QUIET)
  if(NOT GIT_FOUND)
    message(FATAL_ERROR "Cannot build external dependencies without git")
  endif()
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/.git")
    message(FATAL_ERROR "Project is not a git directory")
  endif()
  execute_process(
    COMMAND ${GIT_EXECUTABLE} submodule update --init --remote ${NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND_ERROR_IS_FATAL ANY
  )
endfunction()

# Metal
if(NOT DEFINED TT_BUILD_METAL)
  find_package(Metal QUIET)
  if(${Metal_FOUND})
    set(Metal_NOT_FOUND OFF)
  else()
    set(Metal_NOT_FOUND ON)
  endif()
  option(TT_BUILD_METAL "Build metal dependency" ${Metal_NOT_FOUND})
endif()
if(TT_BUILD_METAL)
  set(METAL_BUILD_DOC OFF CACHE BOOL "")
  set(METAL_BUILD_EXAMPLES OFF CACHE BOOL "")
  set(METAL_BUILD_TESTS OFF CACHE BOOL "")
  git_submodule("metal")
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/metal" "extern/metal")
else()
  find_package(Metal REQUIRED)
endif()
