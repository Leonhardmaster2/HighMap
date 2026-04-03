#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  auto z1 = detrend_reg(z0);

  hmap::export_banner_png("ex_detrend.png", {z0, z1}, hmap::Cmap::JET);
}
