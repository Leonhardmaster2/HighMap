#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  auto d = hmap::laplacian(z);

  // same as standard laplacian
  auto df0 = hmap::gpu::laplacian_fract(z, /* s */ 1.f, /* ir */ 1);

  int   ir = 8;
  float s = 0.2f; // fractional exponent

  auto df1 = hmap::gpu::laplacian_fract(z, s, ir);

  hmap::export_banner_png("ex_laplacian.png",
                          {z, d, df0, df1},
                          hmap::Cmap::JET,
                          true);
}
