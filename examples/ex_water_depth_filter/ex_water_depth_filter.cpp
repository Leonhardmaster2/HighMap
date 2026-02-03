#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::remap(z);

  hmap::Array water_depth0 = hmap::flooding_lake_system(z);
  auto        water_depth1 = water_depth0;

  int ir = 4;
  hmap::gpu::water_depth_filter(water_depth1, z, ir);

  hmap::export_banner_png("ex_water_depth_filter.png",
                          {z, z + water_depth0, z + water_depth1},
                          hmap::Cmap::MAGMA);
}
