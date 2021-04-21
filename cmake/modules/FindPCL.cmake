function(isolated)
  set(FPHSA_NAME_MISMATCHED_OLD ${FPHSA_NAME_MISMATCHED})
  set(FPHSA_NAME_MISMATCHED ON)
  find_package(PCL ${PCL_FIND_VERSION} QUIET CONFIG COMPONENTS ${PCL_FIND_COMPONENTS})
  set(FPHSA_NAME_MISMATCHED ${FPHSA_NAME_MISMATCHED_OLD})

  # Bugfix for: nvcc fatal   : Unknown option 'fPIC'
  # see https://gitlab.kitware.com/cmake/cmake/issues/16915
  list(FILTER PCL_LIBRARIES EXCLUDE REGEX "Qt")

  # Bugfix for: warning: ISO C++11 requires whitespace after the macro name
  # see https://gitlab.kitware.com/vtk/vtk/issues/17361
  set(NEW_COMPILE_DEFINITIONS_PROPERTY)
  get_property(COMPILE_DEFINITIONS_PROPERTY DIRECTORY PROPERTY COMPILE_DEFINITIONS)
  foreach(ITEM ${COMPILE_DEFINITIONS_PROPERTY})
    string(REPLACE "," " ," ITEM ${ITEM})
    string(REPLACE ")" " )" ITEM ${ITEM})
    list(APPEND NEW_COMPILE_DEFINITIONS_PROPERTY ${ITEM})
  endforeach()
  set_property(DIRECTORY PROPERTY COMPILE_DEFINITIONS ${NEW_COMPILE_DEFINITIONS_PROPERTY})

  set(PCL_FOUND ${PCL_FOUND} PARENT_SCOPE)
  set(PCL_VERSION ${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR} PARENT_SCOPE)

  if(PCL_FOUND)
    add_library(PCL::all INTERFACE IMPORTED)
    target_include_directories(PCL::all INTERFACE ${PCL_INCLUDE_DIRS})
    target_compile_definitions(PCL::all INTERFACE ${PCL_DEFINITIONS})
    target_link_libraries(PCL::all INTERFACE ${PCL_LIBRARIES})
  endif()

  foreach(component ${PCL_FIND_COMPONENTS})
    if(PCL_${component}_FOUND)
      add_library(PCL::${component} INTERFACE IMPORTED)
      target_include_directories(PCL::${component} INTERFACE ${PCL_${component}_INCLUDE_DIR})
      target_compile_definitions(PCL::${component} INTERFACE ${PCL_${component}_DEFINITIONS} ${PCL_ADDED_DEFINITIONS})
      target_link_libraries(PCL::${component} INTERFACE ${PCL_${component}_LIBRARY} ${PCL_${component}_LIBRARIES})
    endif()
  endforeach()
endfunction(isolated)
isolated()

if(PCL_FOUND)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(PCL DEFAULT_MSG PCL_VERSION)
endif()
