include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS
    "${PROJECT_SOURCE_DIR}/include/StableHLO/StableHLOTransformOps.td")
mlir_tablegen(StableHLOTransformOps.h.inc -gen-op-decls)
mlir_tablegen(StableHLOTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(StableHLOTransformOpsIncGen)

add_library(NeptuneStableHLO STATIC
  StableHLOLegalizeControlFlow.cpp
  StableHLOTilingInterfaceImpl.cpp
  StableHLOTransformExtension.cpp
  StableHLOTransformOps.cpp
)
add_dependencies(NeptuneStableHLO StableHLOTransformOpsIncGen)
setup_mlir_target(NeptuneStableHLO)
target_link_libraries(NeptuneStableHLO PUBLIC
  NeptuneMLIRCore
  StablehloLinalgTransforms
  StablehloOps
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTilingInterface
  MLIRTransformDialectInterfaces
)
