#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::clamp_min(z0, 0.f);

  auto z1 = z0;
  hmap::equalize(z1);

  hmap::export_banner_png("ex_equalize.png", {z0, z1}, hmap::Cmap::INFERNO);
}
