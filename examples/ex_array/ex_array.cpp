#include <iostream>

#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  std::cout << "min: " << z.min() << "\n";
  std::cout << "max: " << z.max() << "\n";

  glm::vec2 range = z.range();
  std::cout << "range.x: " << range.x << "\n";
  std::cout << "range.y: " << range.y << "\n";

  glm::vec2 range_pc = z.range_percentile(0.05f, 0.95f);
  std::cout << "range_pc.x: " << range_pc.x << "\n";
  std::cout << "range_pc.y: " << range_pc.y << "\n";

  std::cout << "OpenCV build info\n";
  std::cout << hmap::get_opencv_build_information() << "\n";
}
