#include <gtest/gtest.h>

#include "highmap.hpp"

using namespace hmap;

Path helper_generate_path()
{
  const int       seed = 6;
  const glm::vec4 bbox = {0.f, 1.f, 0.f, 1.f};
  const int       npoints = 10;

  Path path = Path(npoints, seed, adjust(bbox, -0.1f));
  path.reorder_nns();
  return path;
}

TEST(PathSplines, PreserveStartEndPoints)
{
  Path path = helper_generate_path();

  EXPECT_EQ(assert_start_end_points(path, bezier(path)), true);
  EXPECT_EQ(assert_start_end_points(path, bezier_round(path)), true);
  EXPECT_EQ(assert_start_end_points(path, bspline(path)), true);
  EXPECT_EQ(assert_start_end_points(path, catmullrom(path)), true);
  EXPECT_EQ(assert_start_end_points(path, decasteljau(path)), true);
}

TEST(PathSplines, PreservePathShape)
{
  Path path = helper_generate_path();

  float tol = 0.15f;

  // NB - not decasteljau, distance too great
  EXPECT_NEAR(chamfer_distance(path, bezier(path)), 0.f, tol);
  EXPECT_NEAR(chamfer_distance(path, bezier_round(path)), 0.f, tol);
  EXPECT_NEAR(chamfer_distance(path, bspline(path)), 0.f, tol);
  EXPECT_NEAR(chamfer_distance(path, catmullrom(path)), 0.f, tol);

  path.set_closed(true);
  EXPECT_NEAR(chamfer_distance(path, bezier(path)), 0.f, tol);
  EXPECT_NEAR(chamfer_distance(path, bezier_round(path)), 0.f, tol);
  EXPECT_NEAR(chamfer_distance(path, bspline(path)), 0.f, tol);
  EXPECT_NEAR(chamfer_distance(path, catmullrom(path)), 0.f, tol);
}

TEST(PathSplines, HasNoDuplicates)
{
  Path path = helper_generate_path();

  EXPECT_EQ(has_duplicates(bezier(path)), false);
  EXPECT_EQ(has_duplicates(bezier_round(path)), false);
  EXPECT_EQ(has_duplicates(bspline(path)), false);
  EXPECT_EQ(has_duplicates(catmullrom(path)), false);
  EXPECT_EQ(has_duplicates(decasteljau(path)), false);

  path.set_closed(true);
  EXPECT_EQ(has_duplicates(bezier(path)), false);
  EXPECT_EQ(has_duplicates(bezier_round(path)), false);
  EXPECT_EQ(has_duplicates(bspline(path)), false);
  EXPECT_EQ(has_duplicates(catmullrom(path)), false);
  EXPECT_EQ(has_duplicates(decasteljau(path)), false);
}
