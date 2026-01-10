#include "highmap.hpp"

int main(void)
{
  hmap::Vec2<int>   shape = {256, 256};
  hmap::Vec2<float> res = {4.f, 4.f};
  int               seed = 0;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z0);

  float riverbed_talus = 0.01f / shape.x;
  auto  z1 = hmap::flow_fixing(z0, riverbed_talus);

  hmap::export_banner_png("ex_flow_fixing.png",
                          {z0, z1},
                          hmap::Cmap::TERRAIN,
                          true);
}
