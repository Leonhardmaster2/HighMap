#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  shape = {1024, 1024};
  glm::vec2     kw = {4.f, 4.f};
  std::uint32_t seed = 0;

  hmap::Array z1 = hmap::gpu::plates(shape, kw, seed, 2.f / shape.x);

  z1.dump();

  hmap::export_banner_png("ex_plates.png", {z1}, hmap::Cmap::TERRAIN, true);
}
