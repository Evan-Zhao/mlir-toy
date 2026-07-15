cmake_minimum_required(VERSION 3.20)

project(NeptuneMLIR LANGUAGES C CXX)

include(GNUInstallDirs)
include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json")

set(_NEPTUNE_MLIR_UNDER_SKBUILD OFF)
if(SKBUILD)
  if(DEFINED SKBUILD_PLATLIB_DIR AND NOT EXISTS "${SKBUILD_PLATLIB_DIR}")
    message(STATUS "Ignoring stale scikit-build cache from ${SKBUILD_PLATLIB_DIR}")
  else()
    set(_NEPTUNE_MLIR_UNDER_SKBUILD ON)
  endif()
endif()

if(_NEPTUNE_MLIR_UNDER_SKBUILD)
  set(NEPTUNE_MLIR_BUILD_PYTHON_EXTENSIONS ON)
else()
  set(NEPTUNE_MLIR_BUILD_PYTHON_EXTENSIONS OFF)
endif()
unset(NEPTUNE_MLIR_BUILD_PYTHON_EXTENSIONS CACHE)
unset(SKBUILD CACHE)

# Find Python3 if not defined. When building from pip, scikit-build will set this variable.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(
  NEPTUNE_MLIR_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}"
  CACHE FILEPATH "Python executable used by the Neptune rolling-solver worker"
)
if(NOT EXISTS "${NEPTUNE_MLIR_PYTHON_EXECUTABLE}")
  message(FATAL_ERROR
    "NEPTUNE_MLIR_PYTHON_EXECUTABLE must name an existing Python executable: "
    "'${NEPTUNE_MLIR_PYTHON_EXECUTABLE}'")
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

if(NEPTUNE_MLIR_BUILD_PYTHON_EXTENSIONS)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import pathlib, mlir._mlir_libs as libs; print(pathlib.Path(libs.__file__).resolve().parent)"
    OUTPUT_VARIABLE MLIR_PYTHON_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _NEPTUNE_MLIR_PYTHON_LIB_QUERY_RESULT
    ERROR_VARIABLE _NEPTUNE_MLIR_PYTHON_LIB_QUERY_ERROR
  )
  if(NOT _NEPTUNE_MLIR_PYTHON_LIB_QUERY_RESULT EQUAL 0)
    message(FATAL_ERROR
      "Failed to locate MLIR Python runtime using Python interpreter "
      "'${Python3_EXECUTABLE}': ${_NEPTUNE_MLIR_PYTHON_LIB_QUERY_ERROR}")
  endif()
endif()

message(STATUS "Using Python3: ${Python3_EXECUTABLE}")
message(STATUS "Using rolling-solver Python: ${NEPTUNE_MLIR_PYTHON_EXECUTABLE}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
if(NEPTUNE_MLIR_BUILD_PYTHON_EXTENSIONS)
  message(STATUS "Using MLIR Python runtime in: ${MLIR_PYTHON_LIB_DIR}")
endif()

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)
include(TableGen)

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

function(setup_mlir_target TARGET)
  llvm_update_compile_flags(${TARGET})
  target_include_directories(${TARGET} PUBLIC
    ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}
  )
  target_include_directories(${TARGET} SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${NEPTUNE_STABLEHLO_INCLUDE_DIRS}
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

# Build StableHLO as a subproject.
set(STABLEHLO_BUILD_EMBEDDED ON)
set(STABLEHLO_ENABLE_BINDINGS_PYTHON OFF)
add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/stablehlo"
  "${CMAKE_CURRENT_BINARY_DIR}/stablehlo"
  EXCLUDE_FROM_ALL
)

add_subdirectory(lib)
add_subdirectory(tools/neptune-opt)
add_subdirectory(tools/neptune-lsp-server)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
