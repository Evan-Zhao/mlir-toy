add_executable(neptune-lsp-server neptune-lsp-server.cpp)
setup_mlir_target(neptune-lsp-server)
target_link_libraries(neptune-lsp-server PRIVATE
  NeptuneInitAll
  MLIRLspServerLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
)

set_target_properties(neptune-lsp-server PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

if(_NEPTUNE_MLIR_UNDER_SKBUILD)
  if(NOT SKBUILD_SCRIPTS_DIR)
    message(FATAL_ERROR "scikit-build did not provide SKBUILD_SCRIPTS_DIR")
  endif()
  install(TARGETS neptune-lsp-server RUNTIME DESTINATION "${SKBUILD_SCRIPTS_DIR}")
else()
  install(TARGETS neptune-lsp-server RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
