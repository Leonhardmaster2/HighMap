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
