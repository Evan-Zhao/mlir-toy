make_mlir_opt_plugin(
  TARGET HTileDialect
  SOURCES HTilePlugin.cpp
  DEPS NeptuneHTile
)
target_link_libraries(HTileDialect PRIVATE NeptuneHTile)

make_mlir_opt_plugin(
  TARGET TADialect
  SOURCES TAPlugin.cpp
  DEPS NeptuneTA
)
target_link_libraries(TADialect PRIVATE NeptuneTA)

make_mlir_opt_plugin(
  TARGET LoopTransform
  SOURCES LoopTrPlugin.cpp
  DEPS NeptuneLoopTransform
)
target_link_libraries(LoopTransform PRIVATE NeptuneLoopTransform)
set_target_properties(LoopTransform PROPERTIES
  BUILD_RPATH "${Python3_LIBRARY_DIRS}"
  INSTALL_RPATH "${Python3_LIBRARY_DIRS}"
)
