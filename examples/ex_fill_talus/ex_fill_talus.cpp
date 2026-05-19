#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::remap(z);

  float talus = 2.f / shape.x;

  hmap::Array z1 = z;
  int         ir = 6;
  float       noise_ratio = 0.01f;
  hmap::fill_talus(z1, talus, seed, ir, noise_ratio);

  // with seed mask
  hmap::Array z2 = z;
  hmap::Array mask = hmap::select_valley(z, 32);
  hmap::saturate(mask, 0.01f, 1.f);

  hmap::fill_talus(z2, talus, seed, ir, noise_ratio, &mask);

  hmap::export_banner_png("ex_fill_talus.png",
                          {z, z1, z2},
                          hmap::Cmap::TERRAIN,
                          true);
}
