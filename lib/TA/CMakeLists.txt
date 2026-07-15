# These are used for tablegen. Mark LLVM/MLIR includes as system headers so
# stricter warning profiles focus on this repo rather than upstream headers.
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS "${PROJECT_SOURCE_DIR}/include/TA/TAOps.td")
mlir_tablegen(TAOps.h.inc -gen-op-decls)
mlir_tablegen(TAOps.cpp.inc -gen-op-defs)
mlir_tablegen(TAOpsDialect.h.inc -gen-dialect-decls -dialect=ta)
mlir_tablegen(TAOpsDialect.cpp.inc -gen-dialect-defs -dialect=ta)
add_public_tablegen_target(TAOpsIncGen)

set(LLVM_TARGET_DEFINITIONS "${PROJECT_SOURCE_DIR}/include/TA/TAAttrs.td")
mlir_tablegen(TAAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=ta)
mlir_tablegen(TAAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ta)
mlir_tablegen(TAEnums.h.inc -gen-enum-decls)
mlir_tablegen(TAEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(TAAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS "${PROJECT_SOURCE_DIR}/include/TA/TATypes.td")
mlir_tablegen(TATypes.h.inc -gen-typedef-decls -typedefs-dialect=ta)
mlir_tablegen(TATypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ta)
add_public_tablegen_target(TATypesIncGen)

set(LLVM_TARGET_DEFINITIONS "${PROJECT_SOURCE_DIR}/include/TA/TATransformOps.td")
mlir_tablegen(TATransformOps.h.inc -gen-op-decls)
mlir_tablegen(TATransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TATransformOpsIncGen)

set(LLVM_TARGET_DEFINITIONS "${PROJECT_SOURCE_DIR}/include/TA/TAPasses.td")
mlir_tablegen(TAPasses.h.inc -gen-pass-decls -name TA)
add_public_tablegen_target(TAPassesIncGen)

add_mlir_pdll_library(TAMulScaleMotionPDLL
  MulScaleMotion.pdll
  MulScaleMotion.cpp.inc
)

add_mlir_pdll_library(TAExpToExp2PDLL
  ExpToExp2.pdll
  ExpToExp2.cpp.inc
)

# Create a wrapper library that takes StablehloOps but make it link to MLIR dynamically.
# There is no StableHLO cmake option that does this.
# This code depends on which targets exist in StableHLO, but it's arguably better than patching
# StableHLO cmake, and we have control over StableHLO version, so all is well.
add_library(NeptuneStablehlo INTERFACE)
add_dependencies(NeptuneStablehlo
  StablehloBase
  StablehloOps
  StablehloTypeInference
  StablehloAssemblyFormat
  StablehloLinalgTransforms
  # Transitive dependencies of StablehloLinalgTransforms:
  StablehloTypeConversion
  StablehloBroadcastLowering
  StablehloBroadcastUtils
  ChloOps                                                                                                                             
)
target_include_directories(NeptuneStablehlo INTERFACE
  "$<TARGET_PROPERTY:StablehloOps,INTERFACE_INCLUDE_DIRECTORIES>"
)
target_link_libraries(NeptuneStablehlo INTERFACE
  "$<TARGET_FILE:StablehloBase>"
  "$<TARGET_FILE:StablehloOps>"
  "$<TARGET_FILE:StablehloTypeInference>"
  "$<TARGET_FILE:StablehloAssemblyFormat>"
  "$<TARGET_FILE:StablehloLinalgTransforms>"
  # Transitive dependencies of StablehloLinalgTransforms:
  "$<TARGET_FILE:StablehloTypeConversion>"
  "$<TARGET_FILE:StablehloBroadcastLowering>"
  "$<TARGET_FILE:StablehloBroadcastUtils>"
  "$<TARGET_FILE:ChloOps>"
  MLIR
  LLVM
)

add_library(NeptuneTA STATIC
  TADialect.cpp
  LinalgToTA.cpp
  StableHLOToTA.cpp
  TAToLinalg.cpp
  TATransformExtension.cpp
  TATransformOps.cpp
  TAUtils.cpp
)
add_dependencies(NeptuneTA
  TAOpsIncGen
  TAAttrsIncGen
  TATypesIncGen
  TAPassesIncGen
  TATransformOpsIncGen
  TAMulScaleMotionPDLL
  TAExpToExp2PDLL
)
setup_mlir_target(NeptuneTA)
target_link_libraries(NeptuneTA PUBLIC NeptuneStablehlo MLIR LLVM)
