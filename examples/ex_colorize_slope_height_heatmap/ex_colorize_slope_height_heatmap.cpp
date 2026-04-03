#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z = hmap::noise(hmap::NoiseType::PERLIN, shape, kw, seed);

  hmap::Tensor col3 = hmap::colorize_slope_height_heatmap(z, hmap::Cmap::JET);
  col3.to_png("ex_colorize_slope_height_heatmap.png");
}
