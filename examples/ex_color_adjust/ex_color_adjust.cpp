#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::remap(z);

  hmap::Texture tex = hmap::colorize(z, 0.f, 1.f, hmap::Cmap::VIRIDIS, false);
  hmap::Array   r = tex[0];
  hmap::Array   g = tex[1];
  hmap::Array   b = tex[2];

  hmap::ColorAdjust params;
  params.contrast = 1.5f;
  params.saturation = 1.2f;

  hmap::color_adjust(r, g, b, params, {0, 0});

  hmap::Texture tex_adj(r, g, b);
  tex_adj.to_png("ex_color_adjust.png");
}
