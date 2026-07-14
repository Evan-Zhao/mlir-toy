add_subdirectory(HTile)
add_subdirectory(TA)
add_subdirectory(LoopTr)
add_subdirectory(Python)

add_library(NeptuneInitAll STATIC InitAll.cpp)
set_target_properties(NeptuneInitAll PROPERTIES POSITION_INDEPENDENT_CODE ON)
setup_mlir_target(NeptuneInitAll)
target_link_libraries(NeptuneInitAll PUBLIC
  NeptuneHTile
  NeptuneTA
  NeptuneLoopTransform
  MLIR
  LLVM
)
