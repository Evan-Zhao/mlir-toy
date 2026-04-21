cmake_minimum_required(VERSION 3.20)

project(NeptuneMLIR LANGUAGES CXX)

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

add_library(HTileDialect STATIC lib/HTile/HTileDialect.cpp)
add_dependencies(HTileDialect HTileOpsIncGen)
llvm_update_compile_flags(HTileDialect)

target_include_directories(HTileDialect
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_compile_definitions(HTileDialect
  PRIVATE
    ${LLVM_DEFINITIONS}
)

mlir_target_link_libraries(HTileDialect
  PUBLIC
    MLIRIR
    MLIRSupport
)

add_executable(hello-mlir hello-mlir.cpp)
llvm_update_compile_flags(hello-mlir)

target_include_directories(hello-mlir
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_compile_definitions(hello-mlir
  PRIVATE
    ${LLVM_DEFINITIONS}
)

mlir_target_link_libraries(hello-mlir
  PRIVATE
    HTileDialect
    MLIRIR
    MLIRPass
    MLIRSupport
)
