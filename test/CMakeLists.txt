configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(NEPTUNE_MLIR_LIT_TEST_DEPENDS
  HTileDialect
  LinalgExtTransform
)

add_lit_testsuite(check-neptune-mlir
  "Running Neptune MLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${NEPTUNE_MLIR_LIT_TEST_DEPENDS}
)
