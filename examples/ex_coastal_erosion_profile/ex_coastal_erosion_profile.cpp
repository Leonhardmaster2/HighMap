#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::SIMPLEX2S, shape, kw, seed);
  hmap::remap(z0);
  auto z1 = z0;
  auto z2 = z0;

  // --- without noise

  hmap::Array water_depth1 = hmap::flooding_uniform_level(z1, 0.3f);

  float shore_ground_extent = 32.f; // pixels
  float shore_water_extent = 16.f;

  hmap::coastal_erosion_profile(z1,
                                water_depth1,
                                shore_ground_extent,
                                shore_water_extent);

  // --- with noise

  hmap::Array dr = 0.5f * hmap::noise_fbm(hmap::NoiseType::SIMPLEX2S,
                                          shape,
                                          4.f * kw,
                                          ++seed,
                                          8,
                                          0.f);

  hmap::Array water_depth2 = hmap::flooding_uniform_level(z2, 0.3f);

  hmap::coastal_erosion_profile(z2,
                                water_depth2,
                                shore_ground_extent,
                                shore_water_extent,
                                0.5f,
                                0.5f,
                                0.5f,
                                true,
                                5,
                                &dr);

  hmap::export_banner_png("ex_coastal_erosion_profile.png",
                          {z0, z1, z1 + water_depth1, z2, z2 + water_depth2},
                          hmap::Cmap::TERRAIN,
                          true);
}
