#include <gtest/gtest.h>

#include "highmap/dbg/assert.hpp"
#include "highmap/local_metrics.hpp"

using namespace hmap;

// ------------------------------------------------------------
// LOCAL MAX
// ------------------------------------------------------------

TEST(LocalMetrics, LocalMax_BasicPeakPropagation)
{
  Array input = Array({{1, 2, 1}, {1, 5, 1}, {1, 1, 1}});

  Array out = local_max(input, 1);

  // center peak should spread locally
  EXPECT_EQ(out(0, 0), 5);
  EXPECT_EQ(out(2, 2), 5);
}

TEST(LocalMetrics, LocalMax_Convergence)
{
  Array input = Array({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

  // fully saturate the array with the max value
  Array a = local_max(local_max(input, 1), 1);
  Array b = local_max(a, 1);

  EXPECT_TRUE(assert_almost_equal(a, b));
}

TEST(LocalMetrics, LocalMax_Monotonicity)
{
  Array input = Array({{1, 3, 2}, {4, 1, 0}, {2, 5, 1}});

  Array out = local_max(input, 1);

  for (int i = 0; i < input.shape.x; ++i)
    for (int j = 0; j < input.shape.y; ++j)
      EXPECT_GE(out(i, j), input(i, j));
}

// ------------------------------------------------------------
// LOCAL MIN
// ------------------------------------------------------------

TEST(LocalMetrics, LocalMin_DualityWithMax)
{
  Array input = Array({{3, 2, 1}, {4, 0, 5}, {6, 7, 8}});

  Array min_via_def = local_min(input, 1);
  Array max_neg = -local_max(-input, 1);

  EXPECT_TRUE(assert_almost_equal(min_via_def, max_neg));
}

TEST(LocalMetrics, LocalMin_Monotonicity)
{
  Array input = Array({{5, 6, 7}, {2, 1, 3}, {4, 8, 9}});

  Array out = local_min(input, 1);

  for (int i = 0; i < input.shape.x; ++i)
    for (int j = 0; j < input.shape.y; ++j)
      EXPECT_LE(out(i, j), input(i, j));
}

// ------------------------------------------------------------
// LOCAL MEAN
// ------------------------------------------------------------

TEST(LocalMetrics, LocalMean_SmoothingEffect)
{
  Array input = Array({{0, 0, 0}, {0, 10, 0}, {0, 0, 0}});

  Array out = local_mean(input, 1);

  // center should decrease due to averaging
  EXPECT_LT(out(1, 1), 10);
}

TEST(LocalMetrics, LocalMean_PreservationOfConstantField)
{
  Array input = Array({{2, 2, 2}, {2, 2, 2}, {2, 2, 2}});

  Array out = local_mean(input, 1);

  EXPECT_TRUE(assert_almost_equal(out, input));
}

TEST(LocalMetrics, LocalMean_Linearity)
{
  Array a = Array({{1, 1, 1}, {1, 1, 1}});

  Array b = Array({{2, 2, 2}, {2, 2, 2}});

  Array mean_a = local_mean(a, 1);
  Array mean_b = local_mean(b, 1);

  for (int i = 0; i < a.shape.x; ++i)
    for (int j = 0; j < a.shape.y; ++j)
      EXPECT_NEAR(mean_b(i, j), 2.f * mean_a(i, j), 1e-5);
}
