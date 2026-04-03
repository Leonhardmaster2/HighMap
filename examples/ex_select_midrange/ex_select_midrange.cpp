#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  float       gain = 1.f;
  hmap::Array c = hmap::select_midrange(z, gain);

  hmap::export_banner_png("ex_select_midrange.png", {z, c}, hmap::Cmap::MAGMA);
}
