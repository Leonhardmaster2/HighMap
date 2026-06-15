#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};

  // --- Base noise for displacement

  glm::vec2     kw = {8.f, 8.f};
  std::uint32_t seed = 0;

  auto dr = 0.2f * hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, ++seed);

  // --- Band

  // raw primitive
  float angle = 30.f;
  float length = 0.4f;
  float width = 0.15f;

  auto z1 = hmap::band(shape, angle, length, width);

  // with edge displacement noise
  auto z2 = hmap::band(shape,
                       angle,
                       length,
                       width,
                       hmap::RadialProfile::RP_SMOOTHSTEP,
                       0.f,
                       &dr);

  // latitude band / polar cap: long band centered on the bottom edge
  auto z3 = hmap::band(shape,
                       0.f,
                       4.f,
                       0.2f,
                       hmap::RadialProfile::RP_SMOOTHSTEP,
                       0.f,
                       &dr,
                       nullptr,
                       {0.5f, 0.f});

  hmap::export_banner_png("ex_band.png", {z1, z2, z3}, hmap::Cmap::TERRAIN, true);
}
