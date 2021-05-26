option(UPDATE_GIT_SUBMODULE "Call 'git update' on required submodules" ON)
function(git_submodule NAME)
  if(UPDATE_GIT_SUBMODULE)
    find_package(Git QUIET)
    if(NOT GIT_FOUND)
      message(FATAL_ERROR "Cannot build external dependencies without git")
    endif()
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/.git")
      message(FATAL_ERROR "Project is not a git directory")
    endif()
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule update --init extern/${NAME}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND_ERROR_IS_FATAL ANY
    )
  endif()
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

# tuple and tuple_utility
git_submodule("tuple")
git_submodule("tuple_utility")
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/extern/tuple/include ${PROJECT_SOURCE_DIR}/extern/tuple/include/jtuple)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/extern/tuple_utility ${PROJECT_SOURCE_DIR}/extern/tuple_utility/jtuple)
add_library(jtuple INTERFACE)
target_include_directories(jtuple
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/extern/tuple/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/extern/tuple_utility>
    $<INSTALL_INTERFACE:include>
)
set(JTUPLE_NAMESPACE_VALUE "jtuple")
set(JTUPLE_NAMESPACE_OPEN_BRACE_VALUE "namespace jtuple \{")
set(JTUPLE_NAMESPACE_CLOSE_BRACE_VALUE "\}")
set(JTUPLE_ANNOTATION "__host__ __device__")
target_compile_definitions(jtuple INTERFACE
  TUPLE_UTILITY_NAMESPACE=${JTUPLE_NAMESPACE_VALUE}
  TUPLE_UTILITY_NAMESPACE_OPEN_BRACE=${JTUPLE_NAMESPACE_OPEN_BRACE_VALUE}
  TUPLE_UTILITY_NAMESPACE_CLOSE_BRACE=${JTUPLE_NAMESPACE_CLOSE_BRACE_VALUE}
  TUPLE_UTILITY_ANNOTATION=${JTUPLE_ANNOTATION}
  TUPLE_NAMESPACE=${JTUPLE_NAMESPACE_VALUE}
  TUPLE_NAMESPACE_OPEN_BRACE=${JTUPLE_NAMESPACE_OPEN_BRACE_VALUE}
  TUPLE_NAMESPACE_CLOSE_BRACE=${JTUPLE_NAMESPACE_CLOSE_BRACE_VALUE}
  TUPLE_ANNOTATION=${JTUPLE_ANNOTATION}
)
install(
  TARGETS jtuple
  EXPORT install_targets
)
install(FILES extern/tuple_utility/tuple_utility.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/jtuple)
install(FILES extern/tuple/include/tuple.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/jtuple)
