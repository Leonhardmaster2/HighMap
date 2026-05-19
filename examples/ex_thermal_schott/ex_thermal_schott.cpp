#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::remap(z);

  auto z1 = z;
  auto z2 = z;

  int   iterations = 500;
  float talus = 2.f / shape.x;

  hmap::gpu::thermal_schott(z1, hmap::Array(shape, talus), iterations);

  // align talus constraint with the elevation
  hmap::Array talus_map = z;
  hmap::remap(talus_map, talus / 2.f, talus);

  hmap::gpu::thermal_schott(z2, talus_map, iterations);

  hmap::export_banner_png("ex_thermal_schott.png",
                          {z, z1, z2},
                          hmap::Cmap::TERRAIN,
                          true);
}
