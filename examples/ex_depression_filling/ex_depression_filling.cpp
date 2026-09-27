#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 0;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  auto z1 = z0;
  hmap::depression_filling(z1);

  auto z2 = z0;
  hmap::depression_filling_priority_flood(z2); // much faster

  // with BC: left / right / bottom / top

  auto z3 = z0;
  hmap::depression_filling(z3, 1000, 1e-4f, false, true, true, true);

  auto z4 = z0;
  hmap::depression_filling_priority_flood(z4,
                                          false /* filter */,
                                          false,
                                          true,
                                          true,
                                          true);

  hmap::export_banner_png("ex_depression_filling.png",
                          {z0, z1, z2, z3, z4},
                          hmap::Cmap::TERRAIN,
                          true);
}
