#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  z += 0.1f * hmap::white(shape, 0.f, 1.f, seed);

  int ir = 16;

  hmap::Array z1 = hmap::median_pseudo(z, ir);

  // gpu version
  hmap::gpu::init_opencl();
  hmap::Array z2 = hmap::gpu::median_pseudo(z, ir);

  hmap::export_banner_png("ex_median_pseudo.png",
                          {z, z1, z2},
                          hmap::Cmap::INFERNO);
}
