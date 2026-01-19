#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  int        seed = 1;

  hmap::Array dmap = hmap::noise_fbm(hmap::NoiseType::PERLIN,
                                     shape,
                                     {2.f, 2.f},
                                     seed);
  hmap::remap(dmap);

  hmap::Array w = hmap::white_density_map(0.5f * dmap, seed);

  hmap::export_banner_png("ex_white_density_map.png",
                          {dmap, w},
                          hmap::Cmap::VIRIDIS);
}
