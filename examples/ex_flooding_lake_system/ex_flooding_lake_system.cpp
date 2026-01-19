#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z);

  hmap::Array water_depth = hmap::flooding_lake_system(z);

  hmap::export_banner_png("ex_flooding_lake_system.png",
                          {z, z + water_depth},
                          hmap::Cmap::JET);
}
