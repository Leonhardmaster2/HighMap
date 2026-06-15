#include <gtest/gtest.h>

#include "highmap.hpp"

TEST(Band, CoreIsOneFalloffToZero)
{
  glm::ivec2 shape = {64, 64};

  // horizontal capsule through the domain center
  hmap::Array z = hmap::band(shape, 0.f, 0.5f, 0.2f);

  EXPECT_NEAR(z(32, 32), 1.f, 1e-6f); // on the core segment
  EXPECT_NEAR(z(0, 0), 0.f, 1e-6f);   // far corner, beyond the falloff
  EXPECT_GE(z.min(), 0.f);
  EXPECT_LE(z.max(), 1.f);
}

TEST(Band, ZeroLengthIsRadialBump)
{
  glm::ivec2 shape = {64, 64};

  hmap::Array z = hmap::band(shape, 0.f, 0.f, 0.3f);

  // point capsule -> radially symmetric: same value at equal distances
  EXPECT_NEAR(z(32 + 10, 32), z(32, 32 + 10), 1e-6f);
  EXPECT_NEAR(z(32 - 10, 32), z(32, 32 - 10), 1e-6f);
}

TEST(Band, LatitudeBandPolarCap)
{
  glm::ivec2 shape = {64, 64};

  // long horizontal band hugging the bottom domain edge -> polar cap
  hmap::Array z = hmap::band(shape,
                             0.f,  // angle
                             4.f,  // length >> domain: infinite strip
                             0.25f,
                             hmap::RadialProfile::RP_SMOOTHSTEP,
                             0.f,
                             nullptr,
                             nullptr,
                             {0.5f, 0.f}); // center on the bottom edge

  for (int i = 0; i < shape.x; i++)
    EXPECT_NEAR(z(i, 0), 1.f, 1e-6f); // whole bottom row on the core

  EXPECT_NEAR(z(32, 40), 0.f, 1e-6f); // mid-domain clear (y = 0.625 > width)
}
