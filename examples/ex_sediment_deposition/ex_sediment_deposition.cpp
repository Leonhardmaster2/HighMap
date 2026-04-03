#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  auto        z0 = z;

  // talus limit defined locally
  auto talus = hmap::Array(shape, 2.f / (float)shape.x);
  hmap::gpu::sediment_deposition(z, talus, nullptr, 0.01f, 10, 50);

  hmap::export_banner_png("ex_sediment_deposition.png",
                          {z0, z},
                          hmap::Cmap::TERRAIN,
                          true);
}
