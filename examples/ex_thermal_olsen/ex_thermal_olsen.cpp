#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2 kw = {4.f, 4.f};
  int       seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::remap(z);
  auto z1 = z;

  float       talus = 2.f / shape.x;
  hmap::Array talus_map = hmap::Array(shape, talus);

  hmap::gpu::thermal_olsen(z1, talus_map, 500);

  hmap::export_banner_png("ex_thermal_olsen.png",
                          {z, z1},
                          hmap::Cmap::TERRAIN,
                          true);
}
