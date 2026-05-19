#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  auto  zs = z;
  float vmin = -0.2f;
  float vmax = 0.2f;
  float k = 0.1f;
  hmap::saturate(zs, vmin, vmax, k);

  auto zp = z;
  // remove 5% lowest and 5% highest
  hmap::saturate_percentile(zp, 0.05f, 0.95f, k);

  hmap::export_banner_png("ex_saturate.png", {z, zs, zp}, hmap::Cmap::MAGMA);
}
