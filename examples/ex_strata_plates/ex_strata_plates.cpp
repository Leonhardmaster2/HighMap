#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);

  hmap::Array talus(shape, 2.f / shape.x);

  auto z1 = z0;
  hmap::gpu::strata_plates(z1, talus);

  hmap::export_banner_png("ex_strata_plates.png",
                          {z0, z1},
                          hmap::Cmap::TERRAIN,
                          true);
}
