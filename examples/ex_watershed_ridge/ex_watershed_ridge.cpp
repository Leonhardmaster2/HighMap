#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 0;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  z0 = hmap::bulkify(z0, hmap::PrimitiveType::PRIM_CONE, 0.5f);

  float amplitude = 0.8f;
  int   ir = 32;
  auto  z1 = hmap::watershed_ridge(z0, amplitude, ir);
  auto  z2 = hmap::gpu::watershed_ridge(z0, amplitude, ir);

  z1.dump("z.png");

  hmap::export_banner_png("ex_watershed_ridge.png",
                          {z0, z1, z2},
                          hmap::Cmap::TERRAIN,
                          true);
}
