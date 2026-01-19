#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 0;

  hmap::Array z = hmap::gpu::basalt_field(shape, kw, seed);

  hmap::export_banner_png("ex_basalt_field.png",
                          {z},
                          hmap::Cmap::INFERNO,
                          true);
}
