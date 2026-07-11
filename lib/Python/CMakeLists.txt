if(NEPTUNE_MLIR_BUILD_PYTHON_EXTENSIONS)
  Python3_add_library(_neptuneMlir MODULE WITH_SOABI
    NeptuneMLIRModule.cpp
    ../HTile/HTileDialect.cpp
    ../HTile/HTilePasses.cpp
    ../HTile/HTileTransformOps.cpp
    ../LoopTr/Utils.cpp
  )
  target_include_directories(_neptuneMlir PRIVATE
    "${PROJECT_BINARY_DIR}/lib/HTile"
    "${PROJECT_BINARY_DIR}/lib/LoopTr"
  )
  target_link_libraries(_neptuneMlir PRIVATE
    "${MLIR_PYTHON_LIB_DIR}/libMLIRPythonCAPI${CMAKE_SHARED_LIBRARY_SUFFIX}"
    MLIR
    LLVM
  )
  set_target_properties(_neptuneMlir PROPERTIES
    BUILD_RPATH "${MLIR_PYTHON_LIB_DIR}"
    INSTALL_RPATH "${MLIR_PYTHON_LIB_DIR}"
  )
  add_dependencies(_neptuneMlir HTileOpsIncGen HTileAttrsIncGen HTileTransformOpsIncGen)
  setup_mlir_target(_neptuneMlir)
  set_target_properties(_neptuneMlir PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/neptune_mlir/_mlir_libs"
  )
  install(TARGETS _neptuneMlir LIBRARY DESTINATION "neptune_mlir/_mlir_libs")
endif()
