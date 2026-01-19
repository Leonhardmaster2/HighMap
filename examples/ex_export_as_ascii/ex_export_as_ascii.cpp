#include <iostream>

#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  glm::ivec2  export_shape = {64, 64};
  std::string str = hmap::export_as_ascii(z, export_shape);

  std::cout << str << "\n";
}
