add_subdirectory(HTile)
add_subdirectory(TA)
add_subdirectory(LoopTr)
add_subdirectory(Python)

add_library(NeptuneInitAll STATIC InitAll.cpp)
setup_mlir_target(NeptuneInitAll)
target_link_libraries(NeptuneInitAll PUBLIC
  NeptuneHTile
  NeptuneTA
  NeptuneLoopTransform
  MLIR
  LLVM
)
