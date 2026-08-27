# HighMap's serial-mode compatibility shim for the NoiseLib submodule.
#
# NoiseLib calls find_package(OpenMP REQUIRED), although its source is valid
# without OpenMP and only uses pragmas for optional parallelism. In serial
# mode, report a deliberately empty interface target. In enabled mode, defer
# to CMake's platform implementation so normal OpenMP discovery is unchanged.

if(HIGHMAP_ENABLE_OPENMP)
  include("${CMAKE_ROOT}/Modules/FindOpenMP.cmake")
else()
  set(OpenMP_FOUND TRUE)
  set(OpenMP_C_FOUND TRUE)
  set(OpenMP_CXX_FOUND TRUE)
  set(OpenMP_C_FLAGS "-Wno-unknown-pragmas")
  set(OpenMP_CXX_FLAGS "-Wno-unknown-pragmas")
  set(OpenMP_C_LIB_NAMES "")
  set(OpenMP_CXX_LIB_NAMES "")

  if(NOT TARGET OpenMP::OpenMP_C)
    add_library(OpenMP::OpenMP_C INTERFACE IMPORTED GLOBAL)
  endif()
  if(NOT TARGET OpenMP::OpenMP_CXX)
    add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED GLOBAL)
  endif()
endif()
