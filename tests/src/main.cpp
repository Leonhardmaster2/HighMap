#include "highmap/opencl/gpu_opencl.hpp"

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

void global_init()
{
  // OpenCL is optional at runtime on macOS. In particular, the system OpenCL
  // framework may be present while no OpenCL device is exposed. The GPU tests
  // that require OpenCL own their availability checks; do not abort the whole
  // test binary during global initialization.
  (void)hmap::gpu::init_opencl();
  spdlog::set_level(spdlog::level::off);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  global_init();

  return RUN_ALL_TESTS();
}
