#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  int         ir = 8;
  hmap::Array w1 = hmap::valley_width(z, ir);

  bool        ridge_select = true;
  hmap::Array w2 = hmap::valley_width(z, ir, ridge_select);

  z.to_png("ex_valley_width0.png", hmap::Cmap::TERRAIN);
  w1.to_png("ex_valley_width1.png", hmap::Cmap::INFERNO);
  w2.to_png("ex_valley_width2.png", hmap::Cmap::INFERNO);
}
