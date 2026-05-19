#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2    shape = {256, 256};
  std::uint32_t seed = 0;

  hmap::Array z1 = hmap::gpu::mountain_tibesti(shape, seed);

  float       scale = 0.5f;
  hmap::Array z2 = hmap::gpu::mountain_tibesti(shape, seed, scale);

  hmap::export_banner_png("ex_mountain_tibesti.png",
                          {z1, z2},
                          hmap::Cmap::TERRAIN,
                          true);
}
