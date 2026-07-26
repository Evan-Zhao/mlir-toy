add_library(NeptuneMLIRCore INTERFACE)
target_link_libraries(NeptuneMLIRCore INTERFACE
  MLIRIR
  MLIRPass
  MLIRTransformDialect
  MLIRTransforms
  LLVMSupport
)

add_subdirectory(TA)
add_subdirectory(LoopTr)
add_subdirectory(HTile)
add_subdirectory(StableHLO)
add_subdirectory(Python)

add_library(NeptuneInitAll STATIC InitAll.cpp)
setup_mlir_target(NeptuneInitAll)
target_link_libraries(NeptuneInitAll PUBLIC
  NeptuneHTile
  NeptuneTA
  NeptuneLoopTransform
  NeptuneStableHLO
  StablehloLinalgTransforms
)
