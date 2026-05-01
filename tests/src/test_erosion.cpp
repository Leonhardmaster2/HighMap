#include <gtest/gtest.h>

#include "highmap/dbg/assert.hpp"
#include "highmap/morphology.hpp"

using namespace hmap;

TEST(Erosion, IdentityWhenRadiusZero)
{
  Array input = Array({{1, 2}, {3, 4}});
  Array cpu = erosion(input, 0);
  Array gpu = gpu::erosion(input, 0);

  EXPECT_TRUE(assert_almost_equal(cpu, input));
  EXPECT_TRUE(assert_almost_equal(gpu, input));
}

TEST(Erosion, SingleMinimumPropagation)
{
  Array input = Array({{5, 5, 5}, {5, 1, 5}, {5, 5, 5}});

  // square kernel for GPU
  Array cpu = erosion(input, 1);
  Array expected_cpu = Array({{1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

  // disk kernel for GPU
  Array gpu = gpu::erosion(input, 1);
  Array expected_gpu = Array({{5, 1, 5}, {1, 1, 1}, {5, 1, 5}});

  EXPECT_TRUE(assert_almost_equal(cpu, expected_cpu));
  EXPECT_TRUE(assert_almost_equal(gpu, expected_gpu));
}

TEST(Erosion, EdgeHandling)
{
  Array input = Array({{3, 3, 3}, {3, 1, 3}, {3, 3, 3}});

  Array cpu = erosion(input, 1);
  Array gpu = gpu::erosion(input, 1);

  EXPECT_EQ(cpu(0, 0), 1);
  EXPECT_EQ(cpu(2, 2), 1);

  EXPECT_EQ(gpu(0, 0), 3);
  EXPECT_EQ(gpu(2, 2), 3);
}

TEST(Erosion, LargerRadius)
{
  Array input = Array({{9, 8, 7, 6}, {8, 5, 4, 7}, {7, 4, 3, 8}, {6, 7, 8, 9}});
  Array expected = Array(
      {{3, 3, 3, 3}, {3, 3, 3, 3}, {3, 3, 3, 3}, {3, 3, 3, 3}});

  Array cpu = erosion(input, 3);
  Array gpu = gpu::erosion(input, 3);

  EXPECT_TRUE(assert_almost_equal(cpu, expected));
  EXPECT_TRUE(assert_almost_equal(gpu, expected));
}

TEST(Erosion, MonotonicDecrease)
{
  Array input = Array({{5, 6}, {7, 8}});

  Array cpu = erosion(input, 1);
  Array gpu = gpu::erosion(input, 1);

  for (int i = 0; i < input.shape.x; ++i)
    for (int j = 0; j < input.shape.y; ++j)
    {
      EXPECT_LE(cpu(i, j), input(i, j));
      EXPECT_LE(gpu(i, j), input(i, j));
    }
}

TEST(Erosion, FlatRegionUnchanged)
{
  Array input = Array({{2, 2, 2}, {2, 2, 2}, {2, 2, 2}});

  Array cpu = erosion(input, 1);
  Array gpu = gpu::erosion(input, 1);

  EXPECT_TRUE(assert_almost_equal(cpu, input));
  EXPECT_TRUE(assert_almost_equal(gpu, input));
}

TEST(Erosion, NonSquareArray)
{
  Array input = Array({{5, 4, 3, 2}});
  Array expected = Array({{4, 3, 2, 2}});

  Array cpu = erosion(input, 1);
  Array gpu = gpu::erosion(input, 1);

  EXPECT_TRUE(assert_almost_equal(cpu, expected));
  EXPECT_TRUE(assert_almost_equal(gpu, expected));
}
