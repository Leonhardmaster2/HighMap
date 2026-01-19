#include <iostream>

#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  std::cout << z.min() << " " << z.max() << std::endl;

  hmap::remap(z, 0.f, 1.f);
  std::cout << z.min() << " " << z.max() << std::endl;
}
