#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);

  int ir = 4;

  hmap::Array c0 = hmap::level_set_curvature(z, ir);
  hmap::Array c1 = hmap::gpu::level_set_curvature(z, ir);

  // --- output

  hmap::export_banner_png("ex_level_set_curvature.png",
                          {z, c0, c1},
                          hmap::Cmap::JET,
                          false,
                          true);
}
