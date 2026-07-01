#include "highmap.hpp"

#include <gtest/gtest.h>

TEST(IslandChain, DeterministicCentersNoNoise)
{
  glm::ivec2 shape = {128, 128};

  hmap::Path path(
      std::vector<hmap::Point>{{0.2f, 0.5f, 0.f}, {0.8f, 0.5f, 0.f}});

  hmap::Array m = hmap::island_chain_land_mask(shape,
                                               path,
                                               1u,    // seed
                                               3,     // island_count
                                               0.08f, // island_radius
                                               0.f,   // size_falloff
                                               0.f,   // size_jitter
                                               0.f,   // scatter
                                               0.f);  // displacement -> discs

  // discs centered at path start / middle / end (x = 0.2, 0.5, 0.8)
  EXPECT_EQ(m(25, 64), 1.f);
  EXPECT_EQ(m(64, 64), 1.f);
  EXPECT_EQ(m(102, 64), 1.f);

  // far off the chain
  EXPECT_EQ(m(64, 16), 0.f);

  EXPECT_GE(m.min(), 0.f);
  EXPECT_LE(m.max(), 1.f);
}

TEST(IslandChain, SizeFalloffShrinksTowardEnd)
{
  glm::ivec2 shape = {128, 128};

  hmap::Path path(
      std::vector<hmap::Point>{{0.2f, 0.5f, 0.f}, {0.8f, 0.5f, 0.f}});

  hmap::Array m = hmap::island_chain_land_mask(shape,
                                               path,
                                               1u,
                                               5,
                                               0.08f,
                                               1.f, // strong falloff to the end
                                               0.f,
                                               0.f,
                                               0.f);

  // land area in the left half (chain head) > right half (chain tail)
  float left = 0.f, right = 0.f;
  for (int j = 0; j < shape.y; j++)
    for (int i = 0; i < shape.x; i++)
      (i < shape.x / 2 ? left : right) += m(i, j);

  EXPECT_GT(left, right);
}
