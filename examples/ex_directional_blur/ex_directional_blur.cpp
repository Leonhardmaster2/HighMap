#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::remap(z);

  float       radius = 64.f;      // pixels
  hmap::Array angle(shape, 30.f); // degrees

  auto z1 = z;
  auto z2 = z; // w/ mask
  auto z3 = z; // variable angle

  hmap::gpu::directional_blur(z1, radius, angle);
  hmap::gpu::directional_blur(z2, radius, angle, &z);

  angle = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, ++seed);
  hmap::remap(angle, -180.f, 180.f);

  hmap::gpu::directional_blur(z3, radius, angle);

  hmap::export_banner_png("ex_directional_blur.png",
                          {z, z1, z2, z3},
                          hmap::Cmap::JET,
                          true);
}
