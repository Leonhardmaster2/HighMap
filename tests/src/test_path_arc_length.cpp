#include <cmath>

#include "highmap.hpp"

#include <gtest/gtest.h>

// Open, non-uniformly spaced path: segments of length 1 then 3.
TEST(PathArcLength, OpenNonUniformCumulativeDistance)
{
  hmap::Path path(std::vector<hmap::Point>{{0.f, 0.f, 0.f},
                                           {1.f, 0.f, 0.f},
                                           {4.f, 0.f, 0.f}});

  std::vector<float> cdist = path.get_cumulative_distance();

  ASSERT_EQ(cdist.size(), 3u);
  EXPECT_NEAR(cdist[0], 0.f, 1e-5f);
  EXPECT_NEAR(cdist[1], 1.f, 1e-5f); // first segment length
  EXPECT_NEAR(cdist[2], 4.f, 1e-5f); // total length

  std::vector<float> arc = path.get_arc_length();
  ASSERT_EQ(arc.size(), 3u);
  EXPECT_NEAR(arc[0], 0.f, 1e-5f);
  EXPECT_NEAR(arc[1], 0.25f, 1e-5f);
  EXPECT_NEAR(arc[2], 1.f, 1e-5f);
}

// Two-point open path: must not collapse to a zero-length (NaN) arc.
TEST(PathArcLength, OpenTwoPointArcLengthIsFinite)
{
  hmap::Path path(
      std::vector<hmap::Point>{{0.2f, 0.5f, 0.f}, {0.8f, 0.5f, 0.f}});

  std::vector<float> cdist = path.get_cumulative_distance();
  ASSERT_EQ(cdist.size(), 2u);
  EXPECT_NEAR(cdist[0], 0.f, 1e-5f);
  EXPECT_NEAR(cdist[1], 0.6f, 1e-5f);

  std::vector<float> arc = path.get_arc_length();
  ASSERT_EQ(arc.size(), 2u);
  EXPECT_TRUE(std::isfinite(arc[0]));
  EXPECT_TRUE(std::isfinite(arc[1]));
  EXPECT_NEAR(arc[0], 0.f, 1e-5f);
  EXPECT_NEAR(arc[1], 1.f, 1e-5f);
}

// Closed unit square: cumulative distance must follow consecutive edges and
// include the closing segment (size N + 1).
TEST(PathArcLength, ClosedSquareCumulativeDistance)
{
  hmap::Path path(std::vector<hmap::Point>{{0.f, 0.f, 0.f},
                                           {1.f, 0.f, 0.f},
                                           {1.f, 1.f, 0.f},
                                           {0.f, 1.f, 0.f}});
  path.set_closed(true);

  std::vector<float> cdist = path.get_cumulative_distance();

  ASSERT_EQ(cdist.size(), 5u);
  EXPECT_NEAR(cdist[0], 0.f, 1e-5f);
  EXPECT_NEAR(cdist[1], 1.f, 1e-5f);
  EXPECT_NEAR(cdist[2], 2.f, 1e-5f);
  EXPECT_NEAR(cdist[3], 3.f, 1e-5f);
  EXPECT_NEAR(cdist[4], 4.f, 1e-5f); // full perimeter
}
