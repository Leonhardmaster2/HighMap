#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z = hmap::gpu::wavelet_noise(shape, kw, seed);

  hmap::export_banner_png("ex_wavelet_noise.png", {z}, hmap::Cmap::JET);
}
