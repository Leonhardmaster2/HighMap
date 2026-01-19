#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z1 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  hmap::Array mask = z1;
  hmap::remap(mask);

  hmap::Array z2 = z1;
  hmap::warp_downslope(z2, &mask, 0.02f, 4, false);

  hmap::export_banner_png("ex_warp_downslope.png",
                          {z1, z2},
                          hmap::Cmap::TERRAIN);
}
