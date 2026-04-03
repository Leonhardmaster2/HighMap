#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise(hmap::NoiseType::PERLIN, shape, res, seed);

  auto zr = z.resample_to_shape({32, 32});

  zr.to_png("ex_resample_to_shape.png", hmap::Cmap::VIRIDIS);
}
