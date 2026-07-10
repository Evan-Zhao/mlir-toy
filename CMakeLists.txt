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
message(STATUS "Using pybind11 CMake package in: ${pybind11_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
if(NEPTUNE_MLIR_BUILD_PYTHON_EXTENSIONS)
  message(STATUS "Using MLIR Python runtime in: ${MLIR_PYTHON_LIB_DIR}")
endif()

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


function(setup_mlir_target TARGET)
  llvm_update_compile_flags(${TARGET})
  target_include_directories(${TARGET} PUBLIC
    ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}
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
  # Keep plugin paths stable when their targets live in subdirectories.
  set_target_properties(${ARG_TARGET} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
  )
  add_dependencies(${ARG_TARGET} ${ARG_DEPS})
  setup_mlir_target(${ARG_TARGET})
  # Do not rely on mlir-opt to provide plugin dependencies through late dynamic
  # lookup. MLIR and LLVM are shared imported targets, so this does not embed
  # duplicate TypeID registries in the plugin.
  target_link_libraries(${ARG_TARGET} PRIVATE MLIR LLVM)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # ELF shared libraries otherwise permit unresolved symbols by default.
    # Reject accidental dependencies on another plugin at link time instead.
    target_link_options(${ARG_TARGET} PRIVATE "LINKER:-z,defs")
  endif()
  install(TARGETS ${ARG_TARGET} LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endfunction()

add_subdirectory(lib/HTile)
add_subdirectory(lib/TA)
add_subdirectory(lib/LoopTr)
add_subdirectory(lib/Python)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
