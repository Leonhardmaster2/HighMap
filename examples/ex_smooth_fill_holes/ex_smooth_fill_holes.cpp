#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {2.f, 2.f};
  int        seed = 1;
  int        radius = 8;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  auto z1 = z;
  hmap::smooth_fill_holes(z1, radius);

  auto z2 = z;
  hmap::smooth_fill_smear_peaks(z2, radius);

  hmap::export_banner_png("ex_smooth_fill_holes.png",
                          {z, z1, z2},
                          hmap::Cmap::TERRAIN,
                          true);
  z.to_file("out.bin");
}
