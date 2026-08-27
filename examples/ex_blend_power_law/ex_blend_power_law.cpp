#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {128, 128};
  int        seed = 1;

  hmap::Array z1 = hmap::noise(hmap::NoiseType::PERLIN,
                               shape,
                               {2.f, 2.f},
                               seed);
  hmap::Array z2 = hmap::noise_fbm(hmap::NoiseType::PERLIN,
                                   shape,
                                   {4.f, 4.f},
                                   seed + 1);

  hmap::remap(z1);
  hmap::remap(z2);

  auto z_alpha0 = hmap::blend_power_law(z1, z2, 0.f);
  auto z_alpha1 = hmap::blend_power_law(z1, z2, 1.f);
  auto z_alpha4 = hmap::blend_power_law(z1, z2, 4.f);
  auto z_alpha16 = hmap::blend_power_law(z1, z2, 16.f);

  hmap::export_banner_png("ex_blend_power_law0.png",
                          {z1, z2, z_alpha0},
                          hmap::Cmap::VIRIDIS);

  hmap::export_banner_png("ex_blend_power_law1.png",
                          {z1, z2, z_alpha1},
                          hmap::Cmap::VIRIDIS);

  hmap::export_banner_png("ex_blend_power_law2.png",
                          {z1, z2, z_alpha4},
                          hmap::Cmap::VIRIDIS);

  hmap::export_banner_png("ex_blend_power_law3.png",
                          {z1, z2, z_alpha16},
                          hmap::Cmap::VIRIDIS);
}
