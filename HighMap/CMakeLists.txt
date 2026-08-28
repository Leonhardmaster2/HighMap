cmake_minimum_required(VERSION 3.22)

project(
  highmap
  VERSION 0.0.1
  LANGUAGES CXX
  DESCRIPTION
    "C++ library to generate two-dimensional terrain heightmaps for software rendering or video games."
)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(${PROJECT_NAME} STATIC ${SOURCES})

if(HIGHMAP_METAL_AVAILABLE)
  # Keep the canonical MSL source next to the backend implementation while
  # embedding it into the library as a generated C++ string. This avoids a
  # runtime dependency on the source checkout and lets the backend compile
  # its pipelines once during initialization.
  set_property(
    DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/gpu_metal/highmap.metal")
  file(READ "${PROJECT_SOURCE_DIR}/src/gpu_metal/highmap.metal"
       HIGHMAP_METAL_SHADER_SOURCE)
  configure_file(
    "${PROJECT_SOURCE_DIR}/src/gpu_metal/metal_shader_source.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/metal_shader_source.hpp"
    @ONLY)

  target_sources(${PROJECT_NAME}
                 PRIVATE ${PROJECT_SOURCE_DIR}/src/gpu_metal/metal_backend.mm)
  target_include_directories(${PROJECT_NAME}
                             PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/gpu_metal/metal_backend.mm
    PROPERTIES COMPILE_OPTIONS "-fobjc-arc")

  if(HIGHMAP_METAL_PRECOMPILED)
    set(HIGHMAP_METAL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/metal")
    set(HIGHMAP_METAL_AIR "${HIGHMAP_METAL_BUILD_DIR}/highmap.air")
    set(HIGHMAP_METALLIB "${HIGHMAP_METAL_BUILD_DIR}/highmap.metallib")
    set(HIGHMAP_METAL_BINARY_HEADER
        "${HIGHMAP_METAL_BUILD_DIR}/metal_library.hpp")

    add_custom_command(
      OUTPUT ${HIGHMAP_METAL_BINARY_HEADER}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${HIGHMAP_METAL_BUILD_DIR}
      COMMAND ${HIGHMAP_XCRUN_EXECUTABLE} --sdk macosx metal -c
              ${PROJECT_SOURCE_DIR}/src/gpu_metal/highmap.metal -o
              ${HIGHMAP_METAL_AIR}
      COMMAND ${HIGHMAP_XCRUN_EXECUTABLE} --sdk macosx metallib
              ${HIGHMAP_METAL_AIR} -o ${HIGHMAP_METALLIB}
      COMMAND ${HIGHMAP_XXD_EXECUTABLE} -i -n highmap_metal_library
              ${HIGHMAP_METALLIB} ${HIGHMAP_METAL_BINARY_HEADER}
      DEPENDS ${PROJECT_SOURCE_DIR}/src/gpu_metal/highmap.metal
      VERBATIM)
    add_custom_target(highmap_metal_library
                      DEPENDS ${HIGHMAP_METAL_BINARY_HEADER})
    add_dependencies(${PROJECT_NAME} highmap_metal_library)
    target_include_directories(${PROJECT_NAME}
                               PRIVATE ${HIGHMAP_METAL_BUILD_DIR})
  endif()

  target_link_libraries(${PROJECT_NAME}
                        PRIVATE ${HIGHMAP_METAL_LIBRARY} "-framework Foundation")
endif()

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC ${PROJECT_SOURCE_DIR}/include
  PRIVATE ${PROJECT_SOURCE_DIR}/include/internal)

set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

target_compile_definitions(
  ${PROJECT_NAME} INTERFACE GLM_FORCE_SWIZZLE GLM_FORCE_INLINE
                            GLM_ENABLE_EXPERIMENTAL _USE_MATH_DEFINES)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC
  assimp::assimp
  delaunator::delaunator
  glm::glm
  GSL::gsl
  GSL::gslcblas
  dkm::dkm
  FastNoiseLite::FastNoiseLite
  hmm::hmm
  libnpy::libnpy
  macro-logger::macro-logger
  nanoflann::nanoflann
  nn-c::nn-c
  mixbox
  NoiseLib
  terrain-descriptors
  ${OpenCV_LIBS}
  point_sampler
  clwrapper
  ${OpenCL_LIBRARIES})

if(HIGHMAP_ENABLE_OPENMP)
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()
