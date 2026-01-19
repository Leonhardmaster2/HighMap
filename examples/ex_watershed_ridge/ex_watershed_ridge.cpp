#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {6.f, 6.f};
  int        seed = 0;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  z0 = hmap::bulkify(z0, hmap::PrimitiveType::PRIM_CONE, 1.f);

  auto z1 = hmap::watershed_ridge(z0);

  hmap::export_banner_png("ex_watershed_ridge.png",
                          {z0, z1},
                          hmap::Cmap::TERRAIN,
                          true);
}
