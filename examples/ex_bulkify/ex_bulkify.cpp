#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  auto zb = hmap::bulkify(z, hmap::PrimitiveType::PRIM_CONE);

  hmap::export_banner_png("ex_bulkify.png", {z, zb}, hmap::Cmap::TERRAIN, true);
}
