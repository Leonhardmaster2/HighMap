#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  auto        z0 = z;

  hmap::Array dmap = hmap::Array(shape);

  hmap::gpu::thermal(z, 0.1f / shape.x, 500);

  hmap::export_banner_png("ex_thermal.png", {z0, z}, hmap::Cmap::TERRAIN, true);
}
