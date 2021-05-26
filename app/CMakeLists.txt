option(BUILD_APPS "Build apps" OFF)

if(BUILD_APPS)
  find_package(nanobench)
  find_package(Boost REQUIRED program_options filesystem)

  if(CMAKE_CUDA_COMPILER)
    if(TARGET nanobench::nanobench)
      add_executable(profile_memcpy profile_memcpy.cu)
      target_link_libraries(profile_memcpy PRIVATE base nanobench::nanobench)
      set_target_properties(profile_memcpy PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/app")

      add_executable(profile_indexstrategy profile_indexstrategy.cpp)
      target_link_libraries(profile_indexstrategy PRIVATE base nanobench::nanobench)
      set_target_properties(profile_indexstrategy PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/app")
    endif()

    if(TARGET interface.opencv AND TARGET interface.nvof)
      add_executable(compute_optical_flow compute_optical_flow.cu)
      target_link_libraries(compute_optical_flow PRIVATE base interface.opencv interface.nvof Boost::program_options Boost::filesystem)
      set_target_properties(compute_optical_flow PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/app")
    endif()

    if(TARGET interface.opencv)
      add_executable(render_skewed render_skewed.cu)
      target_link_libraries(render_skewed PRIVATE base interface.opencv)
      set_target_properties(render_skewed PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/app")
    endif()
  endif()

  # TODO: move profiler into this package? as git submodule? handle all other subpackages (geometry, file, interfaces, etc) the same?
endif()
