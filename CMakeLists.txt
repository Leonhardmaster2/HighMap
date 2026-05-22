cmake_minimum_required(VERSION 3.22)

# -----------------------------------------------------------------------------
# Optional: force clang Usage: cmake -B build -DCMAKE_C_COMPILER=clang
# -DCMAKE_CXX_COMPILER=clang++
# -----------------------------------------------------------------------------

project(highmap-root LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------

option(HIGHMAP_ENABLE_OPENCL "" ON)
option(HIGHMAP_ENABLE_EXAMPLES "" ON)
option(HIGHMAP_ENABLE_TESTS "" ON)
option(HIGHMAP_ENABLE_BENCHMARKS "" ON)

# -----------------------------------------------------------------------------
# C++
# -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# Compiler flags
# -----------------------------------------------------------------------------

add_definitions(-D_USE_MATH_DEFINES)

if(MSVC)

  message(STATUS "Compiler: MSVC")

  add_definitions(-DHMAP_MSVC)

  add_compile_options(/W4 /Od)

  add_compile_definitions(M_PI=3.14159265358979323846)

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

  message(STATUS "Compiler: Clang")

  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -O3
    -Ofast
    -ffast-math
    -pthread
    -fPIC
    -Wno-deprecated-enum-enum-conversion
    -Wno-header-guard
    -Wno-gnu-zero-variadic-macro-arguments)

  add_compile_definitions(LOG_LEVEL=3)

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

  message(STATUS "Compiler: GCC")

  add_compile_options(
    -Wall
    -Wextra
    -O3
    -Ofast
    -ffast-math
    -pthread
    -fPIC
    -Wno-free-nonheap-object
    -Wno-deprecated-enum-enum-conversion)

  add_compile_definitions(LOG_LEVEL=3)

endif()

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

find_package(GSL REQUIRED COMPONENTS gsl gslcblas)
find_package(assimp REQUIRED)
find_package(glm REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# OpenMP
add_compile_options(${OpenMP_CXX_FLAGS})

# OpenCL
add_compile_definitions(CL_HPP_MINIMUM_OPENCL_VERSION=120
                        CL_HPP_TARGET_OPENCL_VERSION=120)

# -----------------------------------------------------------------------------
# External libs
# -----------------------------------------------------------------------------

add_subdirectory(external)

# -----------------------------------------------------------------------------
# Library
# -----------------------------------------------------------------------------

add_subdirectory(HighMap)

# -----------------------------------------------------------------------------
# Examples
# -----------------------------------------------------------------------------

if(HIGHMAP_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

if(HIGHMAP_ENABLE_TESTS)
  message(STATUS "HighMap tests enabled")
  add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------

if(HIGHMAP_ENABLE_BENCHMARKS)
  message(STATUS "HighMap benchmarks enabled")
  add_subdirectory(benchmarks)
endif()
