# These are used for tablegen. Mark LLVM/MLIR includes as system headers so
# stricter warning profiles focus on this repo rather than upstream headers.
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS "${PROJECT_SOURCE_DIR}/include/HTile/HTileOps.td")
mlir_tablegen(HTileOps.h.inc -gen-op-decls)
mlir_tablegen(HTileOps.cpp.inc -gen-op-defs)
mlir_tablegen(HTileOpsDialect.h.inc -gen-dialect-decls -dialect=htile)
mlir_tablegen(HTileOpsDialect.cpp.inc -gen-dialect-defs -dialect=htile)
add_public_tablegen_target(HTileOpsIncGen)

set(LLVM_TARGET_DEFINITIONS "${PROJECT_SOURCE_DIR}/include/HTile/HTileAttrs.td")
mlir_tablegen(HTileAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=htile)
mlir_tablegen(HTileAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=htile)
mlir_tablegen(HTileEnums.h.inc -gen-enum-decls)
mlir_tablegen(HTileEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(HTileAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS "${PROJECT_SOURCE_DIR}/include/HTile/HTileTransformOps.td")
mlir_tablegen(HTileTransformOps.h.inc -gen-op-decls)
mlir_tablegen(HTileTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(HTileTransformOpsIncGen)

set(LLVM_TARGET_DEFINITIONS "${PROJECT_SOURCE_DIR}/include/HTile/HTilePasses.td")
mlir_tablegen(HTilePasses.h.inc -gen-pass-decls -name HTile)
add_public_tablegen_target(HTilePassesIncGen)

add_library(NeptuneHTile STATIC
  HTileDialect.cpp
  HTileOutlineKernels.cpp
  HTilePasses.cpp
  HTileTransformExtension.cpp
  HTileTransformOps.cpp
)
add_dependencies(NeptuneHTile
  HTileOpsIncGen
  HTileAttrsIncGen
  HTilePassesIncGen
  HTileTransformOpsIncGen
)
setup_mlir_target(NeptuneHTile)
target_link_libraries(NeptuneHTile
  PRIVATE
  NeptuneLoopTransform
  PUBLIC
  NeptuneMLIRCore
  StablehloOps
  MLIRAffineUtils
  MLIRArithDialect
  MLIRArithUtils
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRParallelCombiningOpInterface
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTensorUtils
  MLIRTransformDialectInterfaces
  MLIRTransformUtils
)
