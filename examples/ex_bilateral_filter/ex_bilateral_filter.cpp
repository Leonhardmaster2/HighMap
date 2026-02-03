#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);

  int   ir = 16;
  float vscale = 0.01f;

  auto z1 = hmap::gpu::bilateral_filter(z0, ir, vscale);

  hmap::export_banner_png("ex_bilateral_filter.png",
                          {z0, z1},
                          hmap::Cmap::VIRIDIS);
}
