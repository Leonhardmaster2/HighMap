#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2    shape = {256, 256};
  std::uint32_t seed = 3;

  hmap::Array z1 = hmap::gpu::shattered_peak(shape, seed);

  hmap::export_banner_png("ex_shattered_peak.png",
                          {z1},
                          hmap::Cmap::TERRAIN,
                          true);
}
