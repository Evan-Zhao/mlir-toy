add_executable(neptune-opt neptune-opt.cpp)
setup_mlir_target(neptune-opt)
target_link_libraries(neptune-opt PRIVATE NeptuneInitAll MLIR LLVM)
export_executable_symbols_for_plugins(neptune-opt)
set(_NEPTUNE_OPT_INSTALL_RPATH "${LLVM_LIBRARY_DIRS}")
if(APPLE)
  list(PREPEND _NEPTUNE_OPT_INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
  list(PREPEND _NEPTUNE_OPT_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
set_target_properties(neptune-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
  INSTALL_RPATH "${_NEPTUNE_OPT_INSTALL_RPATH}"
)
unset(_NEPTUNE_OPT_INSTALL_RPATH)

if(_NEPTUNE_MLIR_UNDER_SKBUILD)
  if(NOT SKBUILD_SCRIPTS_DIR)
    message(FATAL_ERROR "scikit-build did not provide SKBUILD_SCRIPTS_DIR")
  endif()
  install(TARGETS neptune-opt RUNTIME DESTINATION "${SKBUILD_SCRIPTS_DIR}")
else()
  install(TARGETS neptune-opt RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
