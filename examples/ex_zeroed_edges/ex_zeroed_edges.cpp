#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 0;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::remap(z0);

  auto z1 = z0;
  hmap::zeroed_edges(z1, hmap::RadialProfile::RP_SMOOTHSTEP);

  hmap::export_banner_png("ex_zeroed_edges.png",
                          {z0, z1},
                          hmap::Cmap::TERRAIN,
                          true);
}
