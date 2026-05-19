#include <glm/glm.hpp>

#include "highmap/geometry/point.hpp"

#include <gtest/gtest.h>

using namespace hmap;

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

static constexpr float eps = 1e-5f;

bool float_eq(float a, float b, float tol = eps)
{
  return std::abs(a - b) < tol;
}

// ------------------------------------------------------------
// Constructors
// ------------------------------------------------------------

TEST(PointTest, DefaultConstructor)
{
  Point p;

  EXPECT_TRUE(float_eq(p.x, 0.f));
  EXPECT_TRUE(float_eq(p.y, 0.f));
  EXPECT_TRUE(float_eq(p.v, 0.f));
}

TEST(PointTest, ParameterizedConstructor)
{
  Point p(1.f, 2.f, 3.f);

  EXPECT_TRUE(float_eq(p.x, 1.f));
  EXPECT_TRUE(float_eq(p.y, 2.f));
  EXPECT_TRUE(float_eq(p.v, 3.f));
}

// ------------------------------------------------------------
// Operators
// ------------------------------------------------------------

TEST(PointTest, EqualityOperator)
{
  Point a(1, 2, 3);
  Point b(1, 2, 3);

  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a != b);
}

TEST(PointTest, InequalityOperator)
{
  Point a(1, 2, 3);
  Point b(1, 2, 4);

  EXPECT_TRUE(a != b);
}

TEST(PointTest, AdditionOperator)
{
  Point a(1, 2, 3);
  Point b(4, 5, 6);

  Point c = a + b;

  EXPECT_TRUE(float_eq(c.x, 5));
  EXPECT_TRUE(float_eq(c.y, 7));
  EXPECT_TRUE(float_eq(c.v, 9));
}

TEST(PointTest, SubtractionOperator)
{
  Point a(5, 7, 9);
  Point b(1, 2, 3);

  Point c = a - b;

  EXPECT_TRUE(float_eq(c.x, 4));
  EXPECT_TRUE(float_eq(c.y, 5));
  EXPECT_TRUE(float_eq(c.v, 6));
}

TEST(PointTest, ScalarMultiplicationRight)
{
  Point a(1, 2, 3);

  Point c = a * 2.f;

  EXPECT_TRUE(float_eq(c.x, 2));
  EXPECT_TRUE(float_eq(c.y, 4));
  EXPECT_TRUE(float_eq(c.v, 6));
}

TEST(PointTest, ScalarMultiplicationLeft)
{
  Point a(1, 2, 3);

  Point c = 2.f * a;

  EXPECT_TRUE(float_eq(c.x, 2));
  EXPECT_TRUE(float_eq(c.y, 4));
  EXPECT_TRUE(float_eq(c.v, 6));
}

TEST(PointTest, ScalarDivision)
{
  Point a(2, 4, 6);

  Point c = a / 2.f;

  EXPECT_TRUE(float_eq(c.x, 1));
  EXPECT_TRUE(float_eq(c.y, 2));
  EXPECT_TRUE(float_eq(c.v, 3));
}

// ------------------------------------------------------------
// Geometry
// ------------------------------------------------------------

TEST(PointTest, DistanceBasic)
{
  Point a(0, 0, 0);
  Point b(3, 4, 0);

  EXPECT_TRUE(float_eq(distance(a, b), 5.f));
}

TEST(PointTest, CrossProductOrientation)
{
  Point p0(0, 0, 0);
  Point p1(1, 0, 0);
  Point p2(0, 1, 0);

  float cross = cross_product(p0, p1, p2);

  EXPECT_GT(cross, 0.f); // CCW
}

TEST(PointTest, CrossProductColinear)
{
  Point p0(0, 0, 0);
  Point p1(1, 1, 0);
  Point p2(2, 2, 0);

  EXPECT_TRUE(float_eq(cross_product(p0, p1, p2), 0.f));
}

TEST(PointTest, AngleBetweenPoints)
{
  Point a(0, 0, 0);
  Point b(1, 0, 0);

  float ang = angle(a, b);

  EXPECT_TRUE(float_eq(ang, 0.f));
}

TEST(PointTest, AngleThreePointsRightAngle)
{
  Point p0(0, 0, 0);
  Point p1(1, 0, 0);
  Point p2(0, 1, 0);

  float ang = angle(p0, p1, p2);

  EXPECT_NEAR(std::abs(ang), M_PI / 2.f, 1e-4f);
}

// ------------------------------------------------------------
// Bounding Box
// ------------------------------------------------------------

TEST(PointTest, PointInsideBoundingBox)
{
  Point p(0.5f, 0.5f, 0.f);

  glm::vec4 bbox{0, 1, 0, 1};

  EXPECT_TRUE(is_point_within_bounding_box(p, bbox));
}

TEST(PointTest, PointOutsideBoundingBox)
{
  Point p(2.f, 0.5f, 0.f);

  glm::vec4 bbox{0, 1, 0, 1};

  EXPECT_FALSE(is_point_within_bounding_box(p, bbox));
}

// ------------------------------------------------------------
// Interpolation
// ------------------------------------------------------------

TEST(PointTest, LerpEndpoints)
{
  Point a(0, 0, 0);
  Point b(10, 10, 10);

  Point p0 = lerp(a, b, 0.f);
  Point p1 = lerp(a, b, 1.f);

  EXPECT_TRUE(p0 == a);
  EXPECT_TRUE(p1 == b);
}

TEST(PointTest, LerpMidpoint)
{
  Point a(0, 0, 0);
  Point b(10, 10, 10);

  Point m = lerp(a, b, 0.5f);

  EXPECT_TRUE(float_eq(m.x, 5.f));
  EXPECT_TRUE(float_eq(m.y, 5.f));
  EXPECT_TRUE(float_eq(m.v, 5.f));
}

TEST(PointTest, BezierEndpoints)
{
  Point p0(0, 0, 0);
  Point p1(1, 0, 0);
  Point p2(1, 1, 0);
  Point p3(0, 1, 0);

  EXPECT_TRUE(interp_bezier(p0, p1, p2, p3, 0.f) == p0);
  EXPECT_TRUE(interp_bezier(p0, p1, p2, p3, 1.f) == p3);
}

TEST(PointTest, CatmullRomEndpoints)
{
  Point p0(0, 0, 0);
  Point p1(1, 1, 1);
  Point p2(2, 2, 2);
  Point p3(3, 3, 3);

  EXPECT_TRUE(interp_catmullrom(p0, p1, p2, p3, 0.f) == p1);
  EXPECT_TRUE(interp_catmullrom(p0, p1, p2, p3, 1.f) == p2);
}

// ------------------------------------------------------------
// Segment Intersection
// ------------------------------------------------------------

TEST(PointTest, SegmentIntersectionExists)
{
  Point p1(0, 0, 0), p2(1, 1, 0);
  Point q1(0, 1, 0), q2(1, 0, 0);

  auto result = segment_intersection(p1, p2, q1, q2);

  ASSERT_TRUE(result.has_value());

  EXPECT_NEAR(result->x, 0.5f, eps);
  EXPECT_NEAR(result->y, 0.5f, eps);
}

TEST(PointTest, SegmentIntersectionNone)
{
  Point p1(0, 0, 0), p2(1, 0, 0);
  Point q1(0, 1, 0), q2(1, 1, 0);

  auto result = segment_intersection(p1, p2, q1, q2);

  EXPECT_FALSE(result.has_value());
}

// ------------------------------------------------------------
// Triangle Area
// ------------------------------------------------------------

TEST(PointTest, TriangleAreaBasic)
{
  Point a(0, 0, 0);
  Point b(1, 0, 0);
  Point c(0, 1, 0);

  EXPECT_TRUE(float_eq(triangle_area(a, b, c), 0.5f));
}

TEST(PointTest, TriangleAreaColinear)
{
  Point a(0, 0, 0);
  Point b(1, 1, 0);
  Point c(2, 2, 0);

  EXPECT_TRUE(float_eq(triangle_area(a, b, c), 0.f));
}

// ------------------------------------------------------------
// Sorting
// ------------------------------------------------------------

TEST(PointTest, SortPointsLexicographic)
{
  std::vector<Point> pts = {{2, 1, 0}, {1, 2, 0}, {1, 1, 0}};

  sort_points(pts);

  EXPECT_TRUE(pts[0] == Point(1, 1, 0));
  EXPECT_TRUE(pts[1] == Point(1, 2, 0));
  EXPECT_TRUE(pts[2] == Point(2, 1, 0));
}
