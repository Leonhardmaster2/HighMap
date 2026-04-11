#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;
  int        radius = 10;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::Array zm = hmap::local_mean(z, radius);

  z.to_png("ex_local_mean0.png", hmap::Cmap::VIRIDIS);
  zm.to_png("ex_local_mean1.png", hmap::Cmap::VIRIDIS);
}
