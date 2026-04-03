#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z);

  hmap::Array c = hmap::select_inward_outward_slope(z);
  hmap::remap(c);

  hmap::export_banner_png("ex_select_inward_outward_slope.png",
                          {z, c},
                          hmap::Cmap::MAGMA);
}
