#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;
  float      value = 0.f;
  int        nbuffer = 100;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  hmap::set_borders(z, value, nbuffer);
  z.to_png("ex_set_borders.png", hmap::Cmap::TERRAIN, true);
}
