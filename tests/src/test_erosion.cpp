#include <gtest/gtest.h>

#include "highmap/dbg/assert.hpp"
#include "highmap/morphology.hpp"

using namespace hmap;

TEST(Erosion, IdentityWhenRadiusZero)
{
  Array input = Array({{1, 2}, {3, 4}});
  Array out = erosion(input, 0);

  EXPECT_TRUE(assert_almost_equal(out, input));
}

TEST(Erosion, SingleMinimumPropagation)
{
  // radius = 1 → min in 3x3 neighborhood
  Array input = Array({{5, 5, 5}, {5, 1, 5}, {5, 5, 5}});
  Array expected = Array({{1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
  Array out = erosion(input, 1);

  EXPECT_TRUE(assert_almost_equal(out, expected));
}

TEST(Erosion, EdgeHandling)
{
  Array input = Array({{3, 3, 3}, {3, 1, 3}, {3, 3, 3}});
  Array out = erosion(input, 1);

  // corners should also "see" the center minimum
  EXPECT_EQ(out(0, 0), 1);
  EXPECT_EQ(out(2, 2), 1);
}

TEST(Erosion, LargerRadius)
{
  Array input = Array({{9, 8, 7, 6}, {8, 5, 4, 7}, {7, 4, 3, 8}, {6, 7, 8, 9}});
  Array expected = Array(
      {{3, 3, 3, 3}, {3, 3, 3, 3}, {3, 3, 3, 3}, {3, 3, 3, 3}});

  // radius = 2 → whole array covered → global minimum = 3
  Array out = erosion(input, 2);

  EXPECT_TRUE(assert_almost_equal(out, expected));
}

TEST(Erosion, MonotonicDecrease)
{
  Array input = Array({{5, 6}, {7, 8}});
  Array out = erosion(input, 1);

  // erosion should never increase values
  for (int i = 0; i < input.shape.x; ++i)
    for (int j = 0; j < input.shape.y; ++j)
      EXPECT_LE(out(i, j), input(i, j));
}

TEST(Erosion, FlatRegionUnchanged)
{
  Array input = Array({{2, 2, 2}, {2, 2, 2}, {2, 2, 2}});
  Array out = erosion(input, 1);

  EXPECT_TRUE(assert_almost_equal(out, input));
}

TEST(Erosion, NonSquareArray)
{
  Array input = Array(std::vector<std::vector<float>>{{5, 4, 3, 2}});
  Array expected = Array(std::vector<std::vector<float>>{{4, 3, 2, 2}});
  Array out = erosion(input, 1);

  EXPECT_TRUE(assert_almost_equal(out, expected));
}
