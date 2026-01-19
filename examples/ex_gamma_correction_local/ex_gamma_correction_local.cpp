#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;
  int        ir = 8;
  float      k_smoothing = 0.1f;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z);
  hmap::gamma_correction_local(z, 0.5f, ir, k_smoothing);
  z.to_png("ex_gamma_correction_local.png", hmap::Cmap::VIRIDIS);
}
