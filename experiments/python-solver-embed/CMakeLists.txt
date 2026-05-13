cmake_minimum_required(VERSION 3.20)

project(PythonSolverEmbed LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
execute_process(
  COMMAND "${REPO_ROOT}/.venv/bin/python" -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE VENV_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND "${REPO_ROOT}/.venv/bin/python" -c "import sys; print(sys.base_prefix)"
  OUTPUT_VARIABLE PYTHON_BASE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)

add_executable(python_solver_embed main.cpp)
llvm_update_compile_flags(python_solver_embed)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(python_solver_embed PRIVATE -fexceptions -frtti)
endif()

target_include_directories(python_solver_embed PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_compile_definitions(python_solver_embed PRIVATE
  ${LLVM_DEFINITIONS}
  PY_DEV_PACKAGE_DIR="${REPO_ROOT}/src"
  PY_SITE_PACKAGES="${VENV_SITE_PACKAGES}"
  PYTHON_BASE_PREFIX="${PYTHON_BASE_PREFIX}"
)
target_link_libraries(python_solver_embed PRIVATE
  pybind11::embed
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
)
