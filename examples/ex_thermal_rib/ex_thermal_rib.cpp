#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  auto        z0 = z;

  hmap::gpu::thermal_rib(z, 5);

  z.infos();

  hmap::export_banner_png("ex_thermal_rib.png",
                          {z0, z},
                          hmap::Cmap::TERRAIN,
                          true);
}
