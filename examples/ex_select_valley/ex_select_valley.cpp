#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 2;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z);

  int         ir = 32;
  hmap::Array w1 = hmap::select_valley(z, ir, true);
  hmap::Array w2 = hmap::gpu::select_valley(z, ir, true);
  hmap::Array w3 = hmap::gpu::select_valley(z, ir, false);

  hmap::export_banner_png("ex_select_valley.png",
                          {z, w1, w2, w3},
                          hmap::Cmap::INFERNO);
}
