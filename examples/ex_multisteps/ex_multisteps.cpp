#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  hmap::Vec2<int> shape = {256, 256};

  shape = {1024, 1024};

  auto z1 = hmap::multisteps(shape, 10.f);

  // with built-in default noise
  uint seed = 0;
  auto z2 = hmap::multisteps(shape, 10.f, seed);

  hmap::export_banner_png("ex_multisteps.png", {z1, z2}, hmap::Cmap::INFERNO);
}
