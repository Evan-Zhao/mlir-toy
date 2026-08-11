include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS "${PROJECT_SOURCE_DIR}/include/Tune/TuneTransformOps.td")
mlir_tablegen(TuneTransformOps.h.inc -gen-op-decls)
mlir_tablegen(TuneTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TuneTransformOpsIncGen)

add_library(NeptuneTuneTransform STATIC
  TuneTransformExtension.cpp
  TuneTransformOps.cpp
)
add_dependencies(NeptuneTuneTransform TuneTransformOpsIncGen)
setup_mlir_target(NeptuneTuneTransform)
target_link_libraries(NeptuneTuneTransform PUBLIC
  NeptuneMLIRCore
  MLIRSideEffectInterfaces
  MLIRTransformDialectInterfaces
  MLIRTransformUtils
)
