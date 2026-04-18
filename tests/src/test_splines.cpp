#include <gtest/gtest.h>

#include "highmap.hpp"

TEST(PathSplines, PreserveStartEndPoints)
{
  int        seed = 6;
  glm::vec4  bbox = {0.f, 1.f, 0.f, 1.f};
  int        npoints = 10;
  hmap::Path path = hmap::Path(npoints, seed, hmap::adjust(bbox, -0.1f));
  path.reorder_nns();

  EXPECT_EQ(hmap::assert_start_end_points(path, hmap::bezier(path)), true);
  EXPECT_EQ(hmap::assert_start_end_points(path, hmap::bezier_round(path)),
            true);
  EXPECT_EQ(hmap::assert_start_end_points(path, hmap::bspline(path)), true);
  EXPECT_EQ(hmap::assert_start_end_points(path, hmap::catmullrom(path)), true);
  EXPECT_EQ(hmap::assert_start_end_points(path, hmap::decasteljau(path)), true);
}

TEST(PathSplines, PreservePathShape)
{
  int        seed = 6;
  glm::vec4  bbox = {0.f, 1.f, 0.f, 1.f};
  int        npoints = 10;
  hmap::Path path = hmap::Path(npoints, seed, hmap::adjust(bbox, -0.1f));
  path.reorder_nns();

  float tol = 0.15f;

  // NB - not decasteljau, distance too great
  EXPECT_NEAR(hmap::chamfer_distance(path, hmap::bezier(path)), 0.f, tol);
  EXPECT_NEAR(hmap::chamfer_distance(path, hmap::bezier_round(path)), 0.f, tol);
  EXPECT_NEAR(hmap::chamfer_distance(path, hmap::bspline(path)), 0.f, tol);
  EXPECT_NEAR(hmap::chamfer_distance(path, hmap::catmullrom(path)), 0.f, tol);

  path.set_closed(true);
  EXPECT_NEAR(hmap::chamfer_distance(path, hmap::bezier(path)), 0.f, tol);
  EXPECT_NEAR(hmap::chamfer_distance(path, hmap::bezier_round(path)), 0.f, tol);
  EXPECT_NEAR(hmap::chamfer_distance(path, hmap::bspline(path)), 0.f, tol);
  EXPECT_NEAR(hmap::chamfer_distance(path, hmap::catmullrom(path)), 0.f, tol);
}

TEST(PathSplines, HasNoDuplicates)
{
  int        seed = 6;
  glm::vec4  bbox = {0.f, 1.f, 0.f, 1.f};
  int        npoints = 10;
  hmap::Path path = hmap::Path(npoints, seed, hmap::adjust(bbox, -0.1f));
  path.reorder_nns();

  EXPECT_EQ(hmap::has_duplicates(hmap::bezier(path)), false);
  EXPECT_EQ(hmap::has_duplicates(hmap::bezier_round(path)), false);
  EXPECT_EQ(hmap::has_duplicates(hmap::bspline(path)), false);
  EXPECT_EQ(hmap::has_duplicates(hmap::catmullrom(path)), false);
  EXPECT_EQ(hmap::has_duplicates(hmap::decasteljau(path)), false);

  path.set_closed(true);
  EXPECT_EQ(hmap::has_duplicates(hmap::bezier(path)), false);
  EXPECT_EQ(hmap::has_duplicates(hmap::bezier_round(path)), false);
  EXPECT_EQ(hmap::has_duplicates(hmap::bspline(path)), false);
  EXPECT_EQ(hmap::has_duplicates(hmap::catmullrom(path)), false);
  EXPECT_EQ(hmap::has_duplicates(hmap::decasteljau(path)), false);
}
