#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  auto        z0 = z;

  hmap::thermal_auto_bedrock(z, 0.5f / shape.x, 10);

  hmap::export_banner_png("ex_thermal_auto_bedrock.png",
                          {z0, z},
                          hmap::Cmap::TERRAIN,
                          true);
}
