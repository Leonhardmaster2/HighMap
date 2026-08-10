#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::Array dz = hmap::gradient_norm(z);

  const auto colormap_colors1 = hmap::get_colormap_data(hmap::Cmap::VIRIDIS);
  const auto colormap_colors2 = hmap::get_colormap_data(hmap::Cmap::MAGMA);

  const auto cpos1 = hmap::linspace(0.f, 1.f, colormap_colors1.size());
  const auto cpos2 = hmap::linspace(0.f, 1.f, colormap_colors2.size());

  hmap::Texture tex0 = hmap::colorize_bivariate(z,
                                                dz,
                                                z.range(),
                                                dz.range(),
                                                cpos1,
                                                cpos2,
                                                colormap_colors1,
                                                colormap_colors2,
                                                hmap::MixMethod::MM_LINEAR);

  tex0.to_png("ex_colorize_bivariate0.png");

  hmap::Texture tex1 = hmap::colorize_bivariate(z,
                                                dz,
                                                z.range(),
                                                dz.range(),
                                                cpos1,
                                                cpos2,
                                                colormap_colors1,
                                                colormap_colors2,
                                                hmap::MixMethod::MM_MIXBOX);

  tex1.to_png("ex_colorize_bivariate1.png");
}
