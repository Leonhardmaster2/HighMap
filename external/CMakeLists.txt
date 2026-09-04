include(delaunator.cmake)
include(dkm.cmake)
include(FastNoiseLite.cmake)
include(hmm.cmake)
include(libnpy.cmake)
include(nn-c.cmake)
include(mixbox.cmake)

if(HIGHMAP_ENABLE_OPENCL)
  add_subdirectory(CLWrapper)
endif()
add_subdirectory(PointSampler)
add_subdirectory(Noise/NoiseLib) # dendry
add_subdirectory(terrain-descriptors/lib)

if(NOT TARGET nanoflann) # to avoid duplicate from PointSampler
  add_subdirectory(nanoflann)
endif()
