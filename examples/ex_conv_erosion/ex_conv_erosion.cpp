#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 0;

  shape = {1024, 1024};
  // shape = {256, 256};

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  // z0 = hmap::bulkify(z0, hmap::PrimitiveType::PRIM_CUBIC_PULSE, 4.f);
  z0 = hmap::bulkify(z0, hmap::PrimitiveType::PRIM_CONE_SMOOTH, 2.f);

  auto z1 = z0;

  int iterations = 20;
  hmap::gpu::conv_erosion(z1, seed, iterations);

  z1.dump("out.png");

  hmap::export_banner_png("ex_conv_erosion.png",
                          {z0, z1},
                          hmap::Cmap::TERRAIN,
                          true);
}
