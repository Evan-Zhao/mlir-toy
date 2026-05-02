cmake_minimum_required(VERSION 3.20)

project(NeptuneMLIR LANGUAGES C CXX)

include(GNUInstallDirs)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddLLVM)
include(TableGen)
include(AddMLIR)

set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

if(BUILD_TESTING AND NOT LLVM_EXTERNAL_LIT)
  # Sadly LLVM installation doesn't provide `llvm-lit`.
  # You'll only have `llvm-lit` around if you have an LLVM *build* dir.
  find_program(NEPTUNE_MLIR_LIT NAMES llvm-lit)
  if(NOT NEPTUNE_MLIR_LIT)
    # Try `lit`, which can be installed via `pip install lit`.
    find_program(NEPTUNE_MLIR_LIT NAMES lit)
  endif()
  if(NEPTUNE_MLIR_LIT)
    set(LLVM_EXTERNAL_LIT "${NEPTUNE_MLIR_LIT}" CACHE STRING "Command used to spawn lit" FORCE)
  endif()
endif()

# These are used for tablegen
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS include/HTile/HTileOps.td)
mlir_tablegen(HTileOps.h.inc -gen-op-decls)
mlir_tablegen(HTileOps.cpp.inc -gen-op-defs)
mlir_tablegen(HTileOpsDialect.h.inc -gen-dialect-decls -dialect=htile)
mlir_tablegen(HTileOpsDialect.cpp.inc -gen-dialect-defs -dialect=htile)
add_public_tablegen_target(HTileOpsIncGen)

set(LLVM_TARGET_DEFINITIONS include/HTile/HTileAttrs.td)
mlir_tablegen(HTileAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=htile)
mlir_tablegen(HTileAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=htile)
mlir_tablegen(HTileEnums.h.inc -gen-enum-decls)
mlir_tablegen(HTileEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(HTileAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS include/LinalgExt/LinalgExtTransformOps.td)
mlir_tablegen(LinalgExtTransformOps.h.inc -gen-op-decls)
mlir_tablegen(LinalgExtTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(LinalgExtTransformOpsIncGen)

function(make_mlir_opt_plugin)
  cmake_parse_arguments(ARG "" "TARGET" "SOURCES;DEPS" ${ARGN})
  # Shared plugin for `mlir-opt --load-dialect-plugin=<path>/libHTileDialect.so`
  # (or `.dylib` for macOS).
  # Must NOT statically embed MLIR libs -- mlir-opt already provides them via
  # libMLIR.so, and duplicate TypeID registries cause a segfault on load.
  add_library(${ARG_TARGET} SHARED ${ARG_SOURCES})
  # Resolve MLIR/LLVM symbols from the host process at runtime.
  if(APPLE)
    target_link_options(${ARG_TARGET} PRIVATE -undefined dynamic_lookup)
  endif()

  add_dependencies(${ARG_TARGET} ${ARG_DEPS})
  llvm_update_compile_flags(${ARG_TARGET})
  target_include_directories(${ARG_TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
  )
  target_compile_definitions(${ARG_TARGET} PRIVATE ${LLVM_DEFINITIONS})
endfunction()

make_mlir_opt_plugin(
  TARGET HTileDialect
  SOURCES lib/HTile/HTileDialect.cpp lib/HTile/HTilePasses.cpp
  DEPS HTileOpsIncGen HTileAttrsIncGen
)
make_mlir_opt_plugin(
  TARGET LinalgExtTransform
  SOURCES lib/LinalgExt/LinalgExtTransformOps.cpp
  DEPS LinalgExtTransformOpsIncGen
)

if(SKBUILD)
  set(NEPTUNE_MLIR_PLUGIN_INSTALL_DIR "neptune_mlir/_native")
else()
  set(NEPTUNE_MLIR_PLUGIN_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
endif()

install(
  TARGETS HTileDialect LinalgExtTransform
  LIBRARY DESTINATION "${NEPTUNE_MLIR_PLUGIN_INSTALL_DIR}"
  RUNTIME DESTINATION "${NEPTUNE_MLIR_PLUGIN_INSTALL_DIR}"
  ARCHIVE DESTINATION "${NEPTUNE_MLIR_PLUGIN_INSTALL_DIR}"
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
