#include "highmap.hpp"

int main(void)
{
  hmap::Vec2<int>   shape = {256, 256};
  hmap::Vec2<float> res = {4.f, 4.f};
  int               seed = 0;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  hmap::Array id = hmap::basin_id_priority_flood(z);
  hmap::remap(id);

  hmap::export_banner_png("ex_basin_id.png", {z, id}, hmap::Cmap::JET);
}
