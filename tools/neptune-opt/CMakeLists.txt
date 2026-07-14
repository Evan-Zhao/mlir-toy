add_executable(neptune-opt neptune-opt.cpp)
setup_mlir_target(neptune-opt)
target_link_libraries(neptune-opt PRIVATE NeptuneInitAll MLIR LLVM)
export_executable_symbols_for_plugins(neptune-opt)
set_target_properties(neptune-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

install(TARGETS neptune-opt RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
