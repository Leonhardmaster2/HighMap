#include "highmap.hpp"

int main(void)
{
  hmap::Vec2<int>   shape = {256, 256};
  hmap::Vec2<float> res = {2.f, 2.f};
  int               seed = 1;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z0);

  hmap::Array z1 = z0;

  hmap::Array mask = z0;
  hmap::saturate(mask, 0.2f, 1.f);
  make_binary(mask);
  mask = border(mask, 1);

  float talus = 0.5f / shape.x;
  float noise_ratio = 0.f;
  int   ir = 3;
  hmap::expand_talus(z1, mask, talus, seed, ir, noise_ratio);

  z1.dump();

  hmap::export_banner_png("ex_expand_talus.png",
                          {z0, mask, z1},
                          hmap::Cmap::TERRAIN,
                          true);
}
