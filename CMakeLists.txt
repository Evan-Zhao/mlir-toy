cmake_minimum_required(VERSION 3.20)

project(NeptuneMLIR LANGUAGES C CXX)

include(GNUInstallDirs)
include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json")

if(SKBUILD)
  set(BUILD_PYTHON_EXTENSIONS_DEFAULT ON)
else()
  set(BUILD_PYTHON_EXTENSIONS_DEFAULT OFF)
endif()
option(
  NEPTUNE_MLIR_BUILD_PYTHON_EXTENSIONS
  "Build Python native extensions instead of mlir-opt plugins"
  ${BUILD_PYTHON_EXTENSIONS_DEFAULT}
)

# Find Python3 if not defined. When building from pip, scikit-build will set this variable.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Set pybind11_DIR if not defined, by directly querying the Python interpreter.
if(NOT DEFINED pybind11_DIR)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _NEPTUNE_MLIR_PYBIND11_QUERY_RESULT
    ERROR_VARIABLE _NEPTUNE_MLIR_PYBIND11_QUERY_ERROR
  )
  if(NOT _NEPTUNE_MLIR_PYBIND11_QUERY_RESULT EQUAL 0)
    message(FATAL_ERROR
      "Failed to locate pybind11 using Python interpreter '${Python3_EXECUTABLE}': "
      "${_NEPTUNE_MLIR_PYBIND11_QUERY_ERROR}")
  endif()
endif()
find_package(pybind11 CONFIG REQUIRED)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

if(NEPTUNE_MLIR_BUILD_PYTHON_EXTENSIONS)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import pathlib, torch_mlir._mlir_libs as libs; print(pathlib.Path(libs.__file__).resolve().parent)"
    OUTPUT_VARIABLE TORCH_MLIR_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _TORCH_MLIR_LIB_QUERY_RESULT
  )
  if(NOT _TORCH_MLIR_LIB_QUERY_RESULT EQUAL 0)
    message(WARNING "Could not determine Torch-MLIR library directory, Python extension may fail to load")
  endif()
endif()

message(STATUS "Using Python3: ${Python3_EXECUTABLE}")
message(STATUS "Using pybind11 CMake package in: ${pybind11_DIR}")
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

# These are used for tablegen. Mark LLVM/MLIR includes as system headers so
# stricter warning profiles focus on this repo rather than upstream headers.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

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

set(LLVM_TARGET_DEFINITIONS include/HTile/HTileTransformOps.td)
mlir_tablegen(HTileTransformOps.h.inc -gen-op-decls)
mlir_tablegen(HTileTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(HTileTransformOpsIncGen)

set(LLVM_TARGET_DEFINITIONS include/LoopTr/LoopTransformOps.td)
mlir_tablegen(LoopTransformOps.h.inc -gen-op-decls)
mlir_tablegen(LoopTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(LoopTransformOpsIncGen)

set(LLVM_TARGET_DEFINITIONS include/TA/TAOps.td)
mlir_tablegen(TAOps.h.inc -gen-op-decls)
mlir_tablegen(TAOps.cpp.inc -gen-op-defs)
mlir_tablegen(TAOpsDialect.h.inc -gen-dialect-decls -dialect=ta)
mlir_tablegen(TAOpsDialect.cpp.inc -gen-dialect-defs -dialect=ta)
add_public_tablegen_target(TAOpsIncGen)

set(LLVM_TARGET_DEFINITIONS include/TA/TAAttrs.td)
mlir_tablegen(TAAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=ta)
mlir_tablegen(TAAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ta)
mlir_tablegen(TAEnums.h.inc -gen-enum-decls)
mlir_tablegen(TAEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(TAAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS include/TA/TATypes.td)
mlir_tablegen(TATypes.h.inc -gen-typedef-decls -typedefs-dialect=ta)
mlir_tablegen(TATypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ta)
add_public_tablegen_target(TATypesIncGen)

set(LLVM_TARGET_DEFINITIONS include/TA/TATransformOps.td)
mlir_tablegen(TATransformOps.h.inc -gen-op-decls)
mlir_tablegen(TATransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TATransformOpsIncGen)

add_mlir_pdll_library(TAExchangeDivAndMatmulPDLL
  lib/TA/ExchangeDivAndMatmul.pdll
  ExchangeDivAndMatmul.cpp.inc
)

add_mlir_pdll_library(TAExpToExp2PDLL
  lib/TA/ExpToExp2.pdll
  ExpToExp2.cpp.inc
)

function(setup_mlir_target TARGET)
  # Resolve MLIR/LLVM symbols from the host process at runtime.
  if(APPLE)
    target_link_options(${TARGET} PRIVATE -undefined dynamic_lookup)
  endif()

  llvm_update_compile_flags(${TARGET})
  target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}
  )
  target_include_directories(${TARGET} SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}
  )
  target_compile_definitions(${TARGET} PRIVATE ${LLVM_DEFINITIONS})

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${TARGET} PRIVATE
      -Wall -Wextra -Wpedantic -Wimplicit-fallthrough
      -Wnon-virtual-dtor -Wunused-function -Wno-unused-parameter
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      target_compile_options(${TARGET} PRIVATE
        -Wunneeded-internal-declaration
      )
    endif()
    if(NEPTUNE_MLIR_WARNINGS_AS_ERRORS)
      target_compile_options(${TARGET} PRIVATE -Werror)
    endif()
  elseif(MSVC)
    target_compile_options(${TARGET} PRIVATE /W4)
    if(NEPTUNE_MLIR_WARNINGS_AS_ERRORS)
      target_compile_options(${TARGET} PRIVATE /WX)
    endif()
  endif()
endfunction()

function(make_mlir_opt_plugin)
  cmake_parse_arguments(ARG "" "TARGET" "SOURCES;DEPS" ${ARGN})
  # Shared plugin for `mlir-opt --load-dialect-plugin=<path>/libHTileDialect.so`
  # (or `.dylib` for macOS).
  # Must NOT statically embed MLIR libs -- mlir-opt already provides them via
  # libMLIR.so, and duplicate TypeID registries cause a segfault on load.
  add_library(${ARG_TARGET} SHARED ${ARG_SOURCES})
  add_dependencies(${ARG_TARGET} ${ARG_DEPS})
  setup_mlir_target(${ARG_TARGET})
  install(TARGETS ${ARG_TARGET} LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endfunction()

function(make_mlir_python_extension)
  cmake_parse_arguments(ARG "" "TARGET" "SOURCES;DEPS" ${ARGN})
  Python3_add_library(${ARG_TARGET} MODULE WITH_SOABI ${ARG_SOURCES})

  # The extension resolves MLIR symbols from the Python MLIR runtime loaded by
  # torch_mlir/mlir. Link Torch-MLIR's aggregate CAPI when available so dynamic
  # loaders can find that runtime without relying on global symbol visibility.
  if (TORCH_MLIR_LIB_DIR)
    target_link_libraries(${ARG_TARGET} PRIVATE
      "${TORCH_MLIR_LIB_DIR}/libTorchMLIRAggregateCAPI${CMAKE_SHARED_LIBRARY_SUFFIX}"
      "${TORCH_MLIR_LIB_DIR}/libMLIRPythonSupport-torch_mlir${CMAKE_SHARED_LIBRARY_SUFFIX}"
    )
    set_target_properties(${ARG_TARGET} PROPERTIES
      BUILD_RPATH "${TORCH_MLIR_LIB_DIR}"
      INSTALL_RPATH "${TORCH_MLIR_LIB_DIR}"
    )
  endif()
  add_dependencies(${ARG_TARGET} ${ARG_DEPS})
  setup_mlir_target(${ARG_TARGET})
  set_target_properties(${ARG_TARGET} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/neptune_mlir/_mlir_libs"
  )
  install(TARGETS _neptuneMlir LIBRARY DESTINATION "neptune_mlir/_mlir_libs")
endfunction()

if(NEPTUNE_MLIR_BUILD_PYTHON_EXTENSIONS)
  make_mlir_python_extension(
    TARGET _neptuneMlir
    SOURCES
      lib/Python/NeptuneMLIRModule.cpp
      lib/HTile/HTileDialect.cpp
      lib/HTile/HTilePasses.cpp
      lib/HTile/HTileTransformOps.cpp
      lib/LoopTr/Utils.cpp
    DEPS
      HTileOpsIncGen
      HTileAttrsIncGen
      HTileTransformOpsIncGen
  )
endif()

make_mlir_opt_plugin(
  TARGET HTileDialect
  SOURCES
    lib/HTile/HTileDialect.cpp
    lib/HTile/HTilePasses.cpp
    lib/HTile/HTileTransformOps.cpp
    lib/LoopTr/Utils.cpp
  DEPS
    HTileOpsIncGen
    HTileAttrsIncGen
    HTileTransformOpsIncGen
)
make_mlir_opt_plugin(
  TARGET TADialect
  SOURCES
    lib/TA/TADialect.cpp
    lib/TA/LinalgToTA.cpp
    lib/TA/TAToLinalg.cpp
    lib/TA/TATransformOps.cpp
    lib/TA/TAUtils.cpp
  DEPS
    TAOpsIncGen
    TAAttrsIncGen
    TATypesIncGen
    TATransformOpsIncGen
    TAExchangeDivAndMatmulPDLL
    TAExpToExp2PDLL
)
make_mlir_opt_plugin(
  TARGET LoopTransform
  SOURCES
    lib/LoopTr/FusionIntoProducer.cpp
    lib/LoopTr/FusionRollingUpdate.cpp
    lib/LoopTr/LinalgTransformOps.cpp
    lib/LoopTr/LoopSpecializeDeadTile.cpp
    lib/LoopTr/LoopTransformPlugin.cpp
    lib/LoopTr/PythonSolver.cpp
    lib/LoopTr/ScfTransformOps.cpp
    lib/LoopTr/Utils.cpp
    DEPS LoopTransformOpsIncGen
)

# This is how we call the Python solver from C++.
target_link_libraries(LoopTransform PRIVATE pybind11::embed)
set_source_files_properties(
  lib/LoopTr/PythonSolver.cpp
  PROPERTIES COMPILE_OPTIONS "-frtti;-fexceptions"
)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
