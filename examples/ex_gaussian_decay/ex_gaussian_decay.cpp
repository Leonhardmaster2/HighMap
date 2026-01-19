#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z, -1.f, 1.f);

  float       sigma = 0.5f;
  hmap::Array zg = hmap::gaussian_decay(z, sigma);

  hmap::export_banner_png("ex_gaussian_decay.png",
                          {z, zg},
                          hmap::Cmap::VIRIDIS);
}
