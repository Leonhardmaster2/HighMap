#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::clamp_min(z, 0.f);
  hmap::remap(z);

  float threshold_size = 64.f * 64.f; // pixels * pixels

  hmap::Array zr = hmap::area_remove(z, threshold_size);

  hmap::Array labels = hmap::connected_components(z);

  hmap::export_banner_png("ex_area_remove_fill.png",
                          {z, zr, labels},
                          hmap::Cmap::JET,
                          false,
                          true);
}
