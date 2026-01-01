#include "highmap.hpp"

int main(void)
{
  hmap::Vec2<int>   shape = {256, 256};
  hmap::Vec2<float> res = {2.f, 2.f};
  int               seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z);

  float talus = 2.f / shape.x;

  hmap::Array z1 = z;
  int         ir = 6;
  float       noise_ratio = 0.01f;
  hmap::fill_talus(z1, talus, seed, ir, noise_ratio);

  // same algo on a coarser mesh to spare some computational time
  hmap::Array z2 = z;
  hmap::fill_talus_fast(z2, hmap::Vec2<int>(64, 64), talus, seed);

  // with seed mask
  hmap::Array z3 = z;
  hmap::Array mask = hmap::select_valley(z, 32);
  hmap::saturate(mask, 0.01f, 1.f);

  hmap::fill_talus(z3, talus, seed, ir, noise_ratio, &mask);

  hmap::export_banner_png("ex_fill_talus.png",
                          {z, z1, z2, z3},
                          hmap::Cmap::TERRAIN,
                          true);
}
