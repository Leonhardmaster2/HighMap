#include "highmap.hpp"

int main(void)
{
  glm::ivec2    shape = {256, 256};
  std::uint32_t seed = 1;

  hmap::Path path(std::vector<hmap::Point>{{0.2f, 0.3f, 0.f},
                                           {0.5f, 0.6f, 0.f},
                                           {0.8f, 0.7f, 0.f}});

  // defaults
  auto m1 = hmap::island_chain_land_mask(shape, path, seed);

  // hotspot-style chain: strong size falloff, some scatter
  auto m2 = hmap::island_chain_land_mask(shape,
                                         path,
                                         seed,
                                         7,     // island_count
                                         0.09f, // island_radius
                                         1.f,   // size_falloff
                                         0.3f,  // size_jitter
                                         0.04f, // scatter
                                         0.15f);

  hmap::export_banner_png("ex_island_chain.png", {m1, m2}, hmap::Cmap::GRAY);
}
