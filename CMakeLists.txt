cmake_minimum_required(VERSION 3.20)

project(NeptuneMLIR LANGUAGES C CXX)

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

set(HTILE_DIALECT_SOURCES lib/HTile/HTileDialect.cpp)

# Static lib for linking into executables (e.g. hello-mlir).
add_library(HTileDialect STATIC ${HTILE_DIALECT_SOURCES})
mlir_target_link_libraries(HTileDialect PUBLIC MLIRIR MLIRSupport)

# Shared plugin for `mlir-opt --load-dialect-plugin=<path>/libHTileDialect.dylib`.
# Must NOT statically embed MLIR libs — mlir-opt already provides them via
# libMLIR.dylib, and duplicate TypeID registries cause a segfault on load.
add_library(HTileDialectPlugin SHARED ${HTILE_DIALECT_SOURCES})
# Resolve MLIR/LLVM symbols from the host process at runtime.
if(APPLE)
  target_link_options(HTileDialectPlugin PRIVATE -undefined dynamic_lookup)
endif()

foreach(target HTileDialect HTileDialectPlugin)
  add_dependencies(${target} HTileOpsIncGen HTileAttrsIncGen)
  llvm_update_compile_flags(${target})
  target_include_directories(${target} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
  )
  target_compile_definitions(${target} PRIVATE ${LLVM_DEFINITIONS})
endforeach()

add_executable(hello-mlir hello-mlir.cpp)
llvm_update_compile_flags(hello-mlir)
target_compile_definitions(hello-mlir PRIVATE ${LLVM_DEFINITIONS})
mlir_target_link_libraries(hello-mlir PRIVATE HTileDialect MLIRPass)
