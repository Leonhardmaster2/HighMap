#include <glm/glm.hpp>

#include "highmap/geometry/cloud.hpp"

#include <gtest/gtest.h>

using namespace hmap;

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

static constexpr float eps = 1e-5f;

bool floatEq(float a, float b, float tol = eps)
{
  return std::abs(a - b) < tol;
}

void expectVecNear(const std::vector<float> &a,
                   const std::vector<float> &b,
                   float                     tol = eps)
{
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i)
    EXPECT_NEAR(a[i], b[i], tol);
}

// ------------------------------------------------------------
// Constructors
// ------------------------------------------------------------

TEST(CloudTest, DefaultConstructorEmpty)
{
  Cloud cloud;
  EXPECT_EQ(cloud.size(), 0);
}

TEST(CloudTest, ConstructFromXY)
{
  std::vector<float> x = {0.f, 1.f};
  std::vector<float> y = {2.f, 3.f};

  Cloud cloud(x, y);

  ASSERT_EQ(cloud.size(), 2);
  EXPECT_TRUE(floatEq(cloud.points[0].x, 0.f));
  EXPECT_TRUE(floatEq(cloud.points[0].y, 2.f));
}

TEST(CloudTest, ConstructFromXYV)
{
  std::vector<float> x = {0.f, 1.f};
  std::vector<float> y = {2.f, 3.f};
  std::vector<float> v = {10.f, 20.f};

  Cloud cloud(x, y, v);

  ASSERT_EQ(cloud.size(), 2);
  EXPECT_TRUE(floatEq(cloud.points[1].v, 20.f));
}

TEST(CloudTest, RandomConstructorDeterministic)
{
  Cloud c1(10, 42);
  Cloud c2(10, 42);

  ASSERT_EQ(c1.size(), c2.size());

  for (size_t i = 0; i < c1.size(); ++i)
  {
    EXPECT_NEAR(c1.points[i].x, c2.points[i].x, eps);
    EXPECT_NEAR(c1.points[i].y, c2.points[i].y, eps);
  }
}

// ------------------------------------------------------------
// Basic Operations
// ------------------------------------------------------------

TEST(CloudTest, AddPointIncreasesSize)
{
  Cloud cloud;

  cloud.add_point({1.f, 2.f, 3.f});

  EXPECT_EQ(cloud.size(), 1);
}

TEST(CloudTest, RemovePointReducesSize)
{
  Cloud cloud(std::vector<Point>{{0, 0, 0}, {1, 1, 1}});

  cloud.remove_point(0);

  ASSERT_EQ(cloud.size(), 1);
  EXPECT_TRUE(floatEq(cloud.points[0].x, 1.f));
}

TEST(CloudTest, ClearEmptiesCloud)
{
  Cloud cloud(std::vector<Point>{{0, 0, 0}, {1, 1, 1}});

  cloud.clear();

  EXPECT_EQ(cloud.size(), 0);
}

// ------------------------------------------------------------
// Accessors
// ------------------------------------------------------------

TEST(CloudTest, GetBBoxCorrect)
{
  Cloud cloud(std::vector<Point>{{-1.f, 2.f, 0.f}, {3.f, 4.f, 0.f}});

  glm::vec4 bbox = cloud.get_bbox();

  EXPECT_TRUE(floatEq(bbox.x, -1.f));
  EXPECT_TRUE(floatEq(bbox.y, 3.f));
  EXPECT_TRUE(floatEq(bbox.z, 2.f));
  EXPECT_TRUE(floatEq(bbox.w, 4.f));
}

TEST(CloudTest, GetCenterCorrect)
{
  Cloud cloud(std::vector<Point>{{0.f, 0.f, 0.f}, {2.f, 2.f, 0.f}});

  Point c = cloud.get_center();

  EXPECT_TRUE(floatEq(c.x, 1.f));
  EXPECT_TRUE(floatEq(c.y, 1.f));
}

TEST(CloudTest, GetValuesCorrect)
{
  Cloud cloud(std::vector<Point>{{0, 0, 1}, {1, 1, 2}});

  auto values = cloud.get_values();

  expectVecNear(values, {1.f, 2.f});
}

TEST(CloudTest, GetValuesMinMax)
{
  Cloud cloud(std::vector<Point>{{0, 0, 5}, {1, 1, 2}, {2, 2, 9}});

  EXPECT_TRUE(floatEq(cloud.get_values_min(), 2.f));
  EXPECT_TRUE(floatEq(cloud.get_values_max(), 9.f));
}

TEST(CloudTest, GetXYConcatenation)
{
  Cloud cloud(std::vector<Point>{{1, 2, 0}, {3, 4, 0}});

  auto xy = cloud.get_xy();

  expectVecNear(xy, {1, 2, 3, 4});
}

// ------------------------------------------------------------
// Nearest Point
// ------------------------------------------------------------

TEST(CloudTest, NearestPointCorrect)
{
  Cloud cloud(std::vector<Point>{{0, 0, 0}, {10, 10, 0}});

  size_t idx = cloud.nearest_point({1.f, 1.f});

  EXPECT_EQ(idx, 0);
}

// ------------------------------------------------------------
// Value Operations
// ------------------------------------------------------------

TEST(CloudTest, SetValuesConstant)
{
  Cloud cloud(std::vector<Point>{{0, 0, 1}, {1, 1, 2}});

  cloud.set_values(5.f);

  for (auto &p : cloud.points)
    EXPECT_TRUE(floatEq(p.v, 5.f));
}

TEST(CloudTest, SetValuesVector)
{
  Cloud cloud(std::vector<Point>{{0, 0, 0}, {1, 1, 0}});

  cloud.set_values({3.f, 4.f});

  EXPECT_TRUE(floatEq(cloud.points[0].v, 3.f));
  EXPECT_TRUE(floatEq(cloud.points[1].v, 4.f));
}

// ------------------------------------------------------------
// Remap
// ------------------------------------------------------------

TEST(CloudTest, RemapValuesRange)
{
  Cloud cloud(std::vector<Point>{{0, 0, 0}, {1, 1, 10}});

  cloud.remap_values(0.f, 1.f);

  EXPECT_TRUE(floatEq(cloud.get_values_min(), 0.f));
  EXPECT_TRUE(floatEq(cloud.get_values_max(), 1.f));
}

// ------------------------------------------------------------
// Randomization / Shuffle
// ------------------------------------------------------------

TEST(CloudTest, RandomizeDeterministic)
{
  Cloud c1(std::vector<Point>{{0, 0, 0}, {1, 1, 1}});
  Cloud c2 = c1;

  c1.randomize(42);
  c2.randomize(42);

  for (size_t i = 0; i < c1.size(); ++i)
  {
    EXPECT_NEAR(c1.points[i].x, c2.points[i].x, eps);
    EXPECT_NEAR(c1.points[i].y, c2.points[i].y, eps);
  }
}

TEST(CloudTest, ShuffleDeterministic)
{
  Cloud c1(std::vector<Point>{{0, 0, 0}, {1, 1, 1}});
  Cloud c2 = c1;

  c1.shuffle(1.f, 1.f, 42, 1.f);
  c2.shuffle(1.f, 1.f, 42, 1.f);

  for (size_t i = 0; i < c1.size(); ++i)
  {
    EXPECT_NEAR(c1.points[i].x, c2.points[i].x, eps);
    EXPECT_NEAR(c1.points[i].v, c2.points[i].v, eps);
  }
}

// ------------------------------------------------------------
// Convex Hull
// ------------------------------------------------------------

TEST(CloudTest, ConvexHullSquare)
{
  Cloud cloud(std::vector<Point>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}});

  auto hull = cloud.get_convex_hull();

  EXPECT_EQ(hull.size(), 4);
}

TEST(CloudTest, ConvexHullIgnoresInnerPoint)
{
  Cloud cloud(std::vector<Point>{{0, 0, 0},
                                 {1, 0, 0},
                                 {1, 1, 0},
                                 {0, 1, 0},
                                 {0.5f, 0.5f, 0}});

  auto hull = cloud.get_convex_hull();

  EXPECT_EQ(hull.size(), 4);
}

// ------------------------------------------------------------
// Merge (Free functions)
// ------------------------------------------------------------

TEST(CloudTest, MergeCloudsConcatenates)
{
  Cloud a(std::vector<Point>{{0, 0, 0}});
  Cloud b(std::vector<Point>{{1, 1, 1}});

  Cloud merged = merge_cloud(a, b);

  EXPECT_EQ(merged.size(), 2);
}
