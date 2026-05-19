#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::remap(z0);

  std::vector<float> weights;
  int                ir_min = 4;
  int                ir_max = 64;

  weights = {0.05f, 0.f, 0.f, 0.f, 1.f, 1.f};
  auto z1 = hmap::gpu::spectral_equalizer(z0, weights, ir_min, ir_max);
  hmap::remap(z1); // make it visible

  weights = {1.f, 1.f};
  auto z2 = hmap::gpu::spectral_equalizer(z0, weights, ir_min, ir_max);

  hmap::export_banner_png("ex_spectral_equalizer.png",
                          {z0, z1, z2},
                          hmap::Cmap::INFERNO);
}
