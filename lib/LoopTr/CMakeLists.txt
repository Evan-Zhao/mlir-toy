# These are used for tablegen. Mark LLVM/MLIR includes as system headers so
# stricter warning profiles focus on this repo rather than upstream headers.
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS "${PROJECT_SOURCE_DIR}/include/LoopTr/LoopTransformOps.td")
mlir_tablegen(LoopTransformOps.h.inc -gen-op-decls)
mlir_tablegen(LoopTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(LoopTransformOpsIncGen)

add_library(NeptuneLoopTransform STATIC
  FusionExprSolver.cpp
  FusionIntoProducer.cpp
  FusionSplitKUpdate.cpp
  FusionRollingUpdate.cpp
  LinalgTransformOps.cpp
  LoopSpecializeDeadTile.cpp
  LoopTransformExtension.cpp
  LoopTransformOps.cpp
  ScfLocalizeScratchTensors.cpp
  ScfTransformOps.cpp
  Utils.cpp
)
add_dependencies(NeptuneLoopTransform LoopTransformOpsIncGen)
setup_mlir_target(NeptuneLoopTransform)
target_compile_definitions(NeptuneLoopTransform PRIVATE
  "NEPTUNE_MLIR_PYTHON_EXECUTABLE=\"${NEPTUNE_MLIR_PYTHON_EXECUTABLE}\""
)
target_link_libraries(NeptuneLoopTransform PUBLIC MLIR LLVM)
