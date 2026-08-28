cmake_minimum_required(VERSION 3.22)

# -----------------------------------------------------------------------------
# Optional: force clang Usage: cmake -B build -DCMAKE_C_COMPILER=clang
# -DCMAKE_CXX_COMPILER=clang++
# -----------------------------------------------------------------------------

project(highmap-root LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------

option(HIGHMAP_ENABLE_OPENCL "Enable the legacy OpenCL compatibility backend" ON)
option(HIGHMAP_ENABLE_METAL "Enable the native Metal compute backend on macOS" ON)
option(HIGHMAP_METAL_RUNTIME_COMPILE
       "Allow runtime MSL compilation when no precompiled metallib is available"
       ON)
option(HIGHMAP_ENABLE_OPENMP "Enable OpenMP acceleration when available" ON)
option(HIGHMAP_ENABLE_LOGS "Enable logging" ON)
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

  # do NOT pass /Od here: add_compile_options lands after the per-config flags
  # on the MSVC command line, where it overrides Release's /O2 (last flag wins),
  # silently disabling optimization in every Release build
  add_compile_options(/W4 "$<$<NOT:$<CONFIG:Debug>>:/fp:fast>")

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

endif()

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# Keep the serial OpenMP compatibility shim local to this superproject. The
# NoiseLib submodule asks for OpenMP unconditionally even when its code is
# compiled serially, so its probe must see the same project-level option.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

find_package(GSL REQUIRED COMPONENTS gsl gslcblas)
find_package(assimp REQUIRED)
find_package(glm REQUIRED)
if(HIGHMAP_ENABLE_OPENMP)

  # Apple Clang does not ship an OpenMP runtime. When Homebrew's libomp is
  # installed, FindOpenMP needs the preprocessor flag and include/library
  # locations supplied explicitly. Keep the flags as a CMake list so they are
  # passed as separate compiler arguments to every target, including the
  # external libraries.
  if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_path(HIGHMAP_LIBOMP_INCLUDE_DIR
              omp.h
              HINTS
                /opt/homebrew/opt/libomp/include
                /usr/local/opt/libomp/include)
    find_library(HIGHMAP_LIBOMP_LIBRARY
                 NAMES omp
                 HINTS
                   /opt/homebrew/opt/libomp/lib
                   /usr/local/opt/libomp/lib)

    if(HIGHMAP_LIBOMP_INCLUDE_DIR AND HIGHMAP_LIBOMP_LIBRARY)
      set(OpenMP_C_FLAGS
          "-Xpreprocessor;-fopenmp;-I${HIGHMAP_LIBOMP_INCLUDE_DIR}"
          CACHE STRING "OpenMP C compiler flags" FORCE)
      set(OpenMP_CXX_FLAGS
          "-Xpreprocessor;-fopenmp;-I${HIGHMAP_LIBOMP_INCLUDE_DIR}"
          CACHE STRING "OpenMP CXX compiler flags" FORCE)
      set(OpenMP_C_LIB_NAMES omp CACHE STRING "OpenMP C library names" FORCE)
      set(OpenMP_CXX_LIB_NAMES omp CACHE STRING "OpenMP CXX library names" FORCE)
      set(OpenMP_omp_LIBRARY
          "${HIGHMAP_LIBOMP_LIBRARY}"
          CACHE FILEPATH "OpenMP runtime library" FORCE)
    endif()
  endif()

  find_package(OpenMP QUIET)
  if(NOT OpenMP_CXX_FOUND)
    message(WARNING
            "OpenMP was requested but no usable runtime was found; continuing without OpenMP")
    set(HIGHMAP_ENABLE_OPENMP OFF)
  endif()
endif()

if(HIGHMAP_ENABLE_OPENMP)
  add_compile_definitions(HIGHMAP_HAS_OPENMP=1)
else()
  add_compile_definitions(HIGHMAP_HAS_OPENMP=0)
endif()

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

# Apple SDKs provide the OpenCL framework but not the Khronos C++ wrapper
# header. Homebrew installs the header and C headers as separate packages, so
# locate both explicitly when FindOpenCL only reports the framework library.
find_path(HIGHMAP_OPENCL_CPP_INCLUDE_DIR
          CL/opencl.hpp
          HINTS
            /opt/homebrew/include
            /opt/homebrew/opt/opencl-clhpp-headers/include
            /usr/local/include
            /usr/local/opt/opencl-clhpp-headers/include)
find_path(HIGHMAP_OPENCL_C_INCLUDE_DIR
          CL/cl.h
          HINTS
            /opt/homebrew/include
            /opt/homebrew/opt/opencl-headers/include
            /usr/local/include
            /usr/local/opt/opencl-headers/include)
if(HIGHMAP_OPENCL_CPP_INCLUDE_DIR)
  include_directories(${HIGHMAP_OPENCL_CPP_INCLUDE_DIR})
endif()
if(HIGHMAP_OPENCL_C_INCLUDE_DIR)
  include_directories(${HIGHMAP_OPENCL_C_INCLUDE_DIR})
endif()

# Metal is deliberately detected by framework headers/library rather than by
# a marketing name (M1/M2/M3/...). Command Line Tools installations commonly
# lack the Metal SDK headers, so a normal non-Metal build remains supported.
set(HIGHMAP_METAL_AVAILABLE OFF)
set(HIGHMAP_METAL_PRECOMPILED OFF)
if(APPLE AND HIGHMAP_ENABLE_METAL)
  find_path(HIGHMAP_METAL_INCLUDE_DIR
            Metal.h
            PATH_SUFFIXES Metal.framework/Headers)
  find_library(HIGHMAP_METAL_LIBRARY NAMES Metal)

  if(HIGHMAP_METAL_INCLUDE_DIR AND HIGHMAP_METAL_LIBRARY)
    enable_language(OBJCXX)
    set(HIGHMAP_METAL_AVAILABLE ON)
    add_compile_definitions(HIGHMAP_HAS_METAL=1)
    message(STATUS "Metal backend: enabled (${HIGHMAP_METAL_LIBRARY})")
  else()
    add_compile_definitions(HIGHMAP_HAS_METAL=0)
    message(WARNING
            "Metal was requested but the Metal SDK/framework was not found; continuing without Metal")
  endif()
else()
  add_compile_definitions(HIGHMAP_HAS_METAL=0)
endif()

if(HIGHMAP_METAL_AVAILABLE)
  string(TOLOWER "${CMAKE_BUILD_TYPE}" HIGHMAP_BUILD_TYPE_LOWER)
  find_program(HIGHMAP_XCRUN_EXECUTABLE xcrun)
  find_program(HIGHMAP_XXD_EXECUTABLE xxd)
  if(HIGHMAP_XCRUN_EXECUTABLE AND HIGHMAP_XXD_EXECUTABLE AND
     HIGHMAP_BUILD_TYPE_LOWER STREQUAL "release")
    execute_process(
      COMMAND ${HIGHMAP_XCRUN_EXECUTABLE} --sdk macosx --find metal
      OUTPUT_VARIABLE HIGHMAP_METAL_COMPILER
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    execute_process(
      COMMAND ${HIGHMAP_XCRUN_EXECUTABLE} --sdk macosx --find metallib
      OUTPUT_VARIABLE HIGHMAP_METALLIB_TOOL
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(HIGHMAP_METAL_COMPILER AND HIGHMAP_METALLIB_TOOL)
      set(HIGHMAP_METAL_PRECOMPILED ON)
      add_compile_definitions(HIGHMAP_METAL_PRECOMPILED=1)
      message(STATUS "Metal release library: build-time metallib embedding enabled")
    endif()
  endif()

  if(NOT HIGHMAP_METAL_PRECOMPILED AND NOT HIGHMAP_METAL_RUNTIME_COMPILE)
    message(FATAL_ERROR
            "Metal runtime compilation is disabled, but no build-time metal/metallib toolchain was found")
  endif()
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

# OpenMP flags/runtime are carried by `OpenMP::OpenMP_CXX` on the targets that
# use it. Do not add `OpenMP_CXX_FLAGS` globally: Apple Clang's
# `-Xpreprocessor` would also capture CMake's dependency flags (`-MD`, `-MF`)
# and break otherwise unrelated external targets.

# OpenCL
add_compile_definitions(CL_HPP_MINIMUM_OPENCL_VERSION=120
                        CL_HPP_TARGET_OPENCL_VERSION=120)

# Logging
if(HIGHMAP_ENABLE_LOGS)
  add_compile_definitions(HIGHMAP_ENABLE_LOGS=1)
else()
  add_compile_definitions(HIGHMAP_ENABLE_LOGS=0)
endif()

# -----------------------------------------------------------------------------
# Status output
# -----------------------------------------------------------------------------

message(STATUS "┌─ HighMap configuration ─────────────")
message(STATUS "│ HIGHMAP_ENABLE_OPENCL:              ${HIGHMAP_ENABLE_OPENCL}")
message(STATUS "│ HIGHMAP_ENABLE_LOGS:                ${HIGHMAP_ENABLE_LOGS}")
message(STATUS "│ HIGHMAP_ENABLE_METAL:               ${HIGHMAP_ENABLE_METAL}")
message(STATUS "│ HIGHMAP_METAL_AVAILABLE:             ${HIGHMAP_METAL_AVAILABLE}")
message(STATUS "│ HIGHMAP_METAL_PRECOMPILED:           ${HIGHMAP_METAL_PRECOMPILED}")
message(STATUS "│ HIGHMAP_ENABLE_OPENMP:              ${HIGHMAP_ENABLE_OPENMP}")
message(
  STATUS "│ HIGHMAP_ENABLE_EXAMPLES:            ${HIGHMAP_ENABLE_EXAMPLES}")
message(STATUS "│ HIGHMAP_ENABLE_TESTS:               ${HIGHMAP_ENABLE_TESTS}")
message(
  STATUS "│ HIGHMAP_ENABLE_BENCHMARKS:          ${HIGHMAP_ENABLE_BENCHMARKS}")
message(STATUS "│ CMAKE_CXX_STANDARD:                 ${CMAKE_CXX_STANDARD}")
message(STATUS "│ CMAKE_BUILD_TYPE:                   ${CMAKE_BUILD_TYPE}")
message(STATUS "│ CMAKE_CXX_COMPILER_ID:              ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "└─────────────────────────────────────")

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
