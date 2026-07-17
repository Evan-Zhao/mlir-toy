include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

add_library(NeptuneStableHLO STATIC
  StableHLOTilingInterfaceImpl.cpp
)
setup_mlir_target(NeptuneStableHLO)
target_link_libraries(NeptuneStableHLO PUBLIC
  NeptuneMLIRCore
  StablehloOps
  MLIRTensorDialect
  MLIRTilingInterface
)
