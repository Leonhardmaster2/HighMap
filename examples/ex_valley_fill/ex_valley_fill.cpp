#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 0;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::SIMPLEX2S, shape, res, seed);
  auto        z0 = z;

  int iterations = 250;
  hmap::gpu::valley_fill(z, hmap::Array(shape, 2.f / shape.x), iterations);

  hmap::export_banner_png("ex_valley_fill.png",
                          {z0, z},
                          hmap::Cmap::TERRAIN,
                          true);
}
