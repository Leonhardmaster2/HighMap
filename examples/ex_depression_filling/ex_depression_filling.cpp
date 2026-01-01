#include "highmap.hpp"

int main(void)
{
  hmap::Vec2<int>   shape = {256, 256};
  hmap::Vec2<float> res = {4.f, 4.f};
  int               seed = 0;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  auto z1 = z0;
  hmap::depression_filling(z1);

  auto z2 = z0;
  hmap::depression_filling_priority_flood(z2); // much faster

  hmap::export_banner_png("ex_depression_filling.png",
                          {z0, z1, z2},
                          hmap::Cmap::TERRAIN,
                          true);
}
