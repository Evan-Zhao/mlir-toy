add_executable(neptune-opt neptune-opt.cpp)
setup_mlir_target(neptune-opt)
target_link_libraries(neptune-opt PRIVATE
  NeptuneInitAll
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
)
set_target_properties(neptune-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

if(_NEPTUNE_MLIR_UNDER_SKBUILD)
  if(NOT SKBUILD_SCRIPTS_DIR)
    message(FATAL_ERROR "scikit-build did not provide SKBUILD_SCRIPTS_DIR")
  endif()
  install(TARGETS neptune-opt RUNTIME DESTINATION "${SKBUILD_SCRIPTS_DIR}")
else()
  install(TARGETS neptune-opt RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
