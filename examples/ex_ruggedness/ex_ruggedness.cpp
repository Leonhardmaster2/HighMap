#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;
  int        ir = 5;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::Array c = hmap::ruggedness(z, ir);

  z.to_png("ex_ruggedness0.png", hmap::Cmap::INFERNO);
  c.to_png("ex_ruggedness1.png", hmap::Cmap::INFERNO);
}
