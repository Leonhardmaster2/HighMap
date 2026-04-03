#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  std::vector<float> k(32);
  std::fill(k.begin(), k.end(), 1.f / 32.f);

  auto zc = hmap::convolve1d_i(z, k);
  zc = hmap::convolve1d_j(zc, k);

  hmap::export_banner_png("ex_convolve1d_ij.png", {z, zc}, hmap::Cmap::VIRIDIS);
}
