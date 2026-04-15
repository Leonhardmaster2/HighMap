#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  shape = {1024, 1024};
  int seed = 0;

  float scale = 1.f / 128.f;

  hmap::Array z = hmap::diffusion_limited_aggregation(shape, scale, seed);

  hmap::Array zt = hmap::diffusion_limited_aggregation_trimesh(shape,
                                                               seed,
                                                               50000);

  hmap::export_banner_png("ex_diffusion_limited_aggregation.png",
                          {z, zt},
                          hmap::Cmap::JET,
                          true);
}
