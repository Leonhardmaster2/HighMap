#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  // shape = {128, 128};
  // shape = {1024, 1024};
  glm::vec2 res = {4.f, 4.f};
  int       seed = 0;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  // hmap::clamp_min(z0, 0.f);
  z0 = hmap::bulkify(z0, hmap::PrimitiveType::PRIM_CUBIC_PULSE, 1.f);
  hmap::remap(z0);

  auto z1 = z0;
  hmap::hydraulic_spl(z1);

  z0.dump("out0.png");
  z1.dump();

  // hmap::remap(z1);

  hmap::export_banner_png("ex_hydraulic_spl.png",
                          {z0, z1},
                          hmap::Cmap::TERRAIN,
                          true);
}
