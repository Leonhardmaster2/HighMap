#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2    shape = {256, 256};
  glm::vec2     kw = {2.f, 2.f};
  std::uint32_t seed = 0;

  hmap::Array z1 = hmap::gpu::badlands(shape, kw, seed);

  hmap::export_banner_png("ex_badlands.png", {z1}, hmap::Cmap::TERRAIN, true);
}
