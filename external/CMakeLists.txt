include(delaunator.cmake)
include(dkm.cmake)
include(FastNoiseLite.cmake)
include(hmm.cmake)
include(libnpy.cmake)
include(macro-logger.cmake)
include(nn-c.cmake)

add_subdirectory(CLWrapper)
add_subdirectory(PointSampler)
add_subdirectory(Noise/NoiseLib) # dendry

if(NOT nanoflann_FOUND)
  add_subdirectory(nanoflann)
endif()
