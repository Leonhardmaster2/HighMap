#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z);

  int ir = 16;

  // gpu version
  hmap::gpu::init_opencl();
  hmap::Array z1 = hmap::gpu::local_z_score(z, ir);
  hmap::remap(z1);

  hmap::export_banner_png("ex_local_z_score.png", {z, z1}, hmap::Cmap::JET);
}
