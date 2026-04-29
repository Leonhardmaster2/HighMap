#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 0;

  hmap::Array r = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::Array g = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, ++seed);
  hmap::Array b = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, ++seed);

  hmap::remap(r);
  hmap::remap(g);
  hmap::remap(b);

  glm::vec3 color = {1.f, 0.f, 0.f}; // red
  float     tolerance = 0.5f;

  hmap::Array mask = hmap::color_match_mask(r, g, b, color, tolerance);

  hmap::export_banner_png("ex_color_match_mask.png",
                          {r, g, b, mask},
                          hmap::Cmap::INFERNO);
}
