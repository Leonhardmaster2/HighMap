#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  auto        z0 = z;

  hmap::Array dmap = hmap::Array(shape);

  hmap::thermal(z, 0.1f / shape.x);

  hmap::export_banner_png("ex_thermal.png", {z0, z}, hmap::Cmap::TERRAIN, true);
}
