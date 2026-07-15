add_executable(neptune-lsp-server neptune-lsp-server.cpp)
setup_mlir_target(neptune-lsp-server)
target_link_libraries(neptune-lsp-server PRIVATE NeptuneInitAll MLIR LLVM)

set(_NEPTUNE_LSP_INSTALL_RPATH "${LLVM_LIBRARY_DIRS}")
if(APPLE)
  list(PREPEND _NEPTUNE_LSP_INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
  list(PREPEND _NEPTUNE_LSP_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
set_target_properties(neptune-lsp-server PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
  INSTALL_RPATH "${_NEPTUNE_LSP_INSTALL_RPATH}"
)
unset(_NEPTUNE_LSP_INSTALL_RPATH)

if(_NEPTUNE_MLIR_UNDER_SKBUILD)
  if(NOT SKBUILD_SCRIPTS_DIR)
    message(FATAL_ERROR "scikit-build did not provide SKBUILD_SCRIPTS_DIR")
  endif()
  install(TARGETS neptune-lsp-server RUNTIME DESTINATION "${SKBUILD_SCRIPTS_DIR}")
else()
  install(TARGETS neptune-lsp-server RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
