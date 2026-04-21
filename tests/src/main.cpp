#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include "highmap/opencl/gpu_opencl.hpp"

void global_init()
{
  hmap::gpu::init_opencl();
  spdlog::set_level(spdlog::level::off);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  global_init();

  return RUN_ALL_TESTS();
}
