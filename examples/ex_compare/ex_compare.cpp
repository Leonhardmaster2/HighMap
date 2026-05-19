#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 1;

  hmap::Array a = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, ++seed);
  hmap::Array b = hmap::noise_fbm(hmap::NoiseType::PERLIN,
                                  shape,
                                  0.5f * kw,
                                  ++seed);

  float       slice_x_pos = 0.5f;
  float       slice_y_pos = 0.2f;
  hmap::Array out = hmap::compare(a, b, slice_x_pos, slice_y_pos);

  hmap::export_banner_png("ex_compare.png", {a, b, out}, hmap::Cmap::INFERNO);
}
