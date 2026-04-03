#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z);

  float talus_ref = 4.f;
  float clipping_ratio = 50.f;

  hmap::Array c = hmap::select_rivers(z, talus_ref, clipping_ratio);
  hmap::remap(c);

  hmap::export_banner_png("ex_select_rivers.png", {z, c}, hmap::Cmap::INFERNO);
}
