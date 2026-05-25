#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};

  // --- Base noise for displacement

  glm::vec2     kw = {8.f, 8.f};
  std::uint32_t seed = 0;

  auto dr = 0.2f * hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, ++seed);
  auto ds = 1.8f * hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, ++seed);

  // --- Rift

  // raw primitive
  float angle = 15.f;

  auto z1 = hmap::rift(shape, angle);

  // with some noises
  float radius = 0.1f;
  float axial_slope = 0.2f;
  float depth = 0.5f;
  bool  scale_with_depth = true;
  auto  profile = hmap::RadialProfile::RP_SMOOTHSTEP;
  float profile_param = 0.f;
  float bottom_extent = 0.1;
  float bottom_depth = 0.05f;
  auto  bottom_profile = hmap::RadialProfile::RP_SQRT;
  float bottom_profile_param = 0.f;
  float outer_slope = 0.5f;

  auto z2 = hmap::rift(shape,
                       angle,
                       radius,
                       axial_slope,
                       depth,
                       scale_with_depth,
                       profile,
                       profile_param,
                       bottom_extent,
                       bottom_depth,
                       bottom_profile,
                       bottom_profile_param,
                       outer_slope,
                       &dr,
                       &ds);

  hmap::export_banner_png("ex_rift.png", {z1, z2}, hmap::Cmap::TERRAIN, true);
}
