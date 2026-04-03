#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  auto zf = z;
  int  order = 7;
  hmap::low_pass_high_order(zf, order);

  hmap::export_banner_png("ex_low_pass_high_order.png",
                          {z, zf},
                          hmap::Cmap::VIRIDIS);
}
