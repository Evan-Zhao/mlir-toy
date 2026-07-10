# These are used for tablegen. Mark LLVM/MLIR includes as system headers so
# stricter warning profiles focus on this repo rather than upstream headers.
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS "${PROJECT_SOURCE_DIR}/include/LoopTr/LoopTransformOps.td")
mlir_tablegen(LoopTransformOps.h.inc -gen-op-decls)
mlir_tablegen(LoopTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(LoopTransformOpsIncGen)

make_mlir_opt_plugin(
  TARGET LoopTransform
  SOURCES
    FusionExprSolver.cpp
    FusionIntoProducer.cpp
    FusionSplitKUpdate.cpp
    FusionRollingUpdate.cpp
    LinalgTransformOps.cpp
    LoopSpecializeDeadTile.cpp
    LoopTransformPlugin.cpp
    ScfLocalizeScratchTensors.cpp
    ScfTransformOps.cpp
    Utils.cpp
  DEPS
    LoopTransformOpsIncGen
)

# This is how we call the Python solver from C++.
target_link_libraries(LoopTransform PRIVATE pybind11::embed)
set_target_properties(LoopTransform PROPERTIES
  BUILD_RPATH "${Python3_LIBRARY_DIRS}"
  INSTALL_RPATH "${Python3_LIBRARY_DIRS}"
)
set_source_files_properties(
  FusionExprSolver.cpp
  PROPERTIES COMPILE_OPTIONS "-frtti;-fexceptions"
)
