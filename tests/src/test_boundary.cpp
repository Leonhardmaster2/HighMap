#include <gtest/gtest.h>
#include "highmap/boundary.hpp"

TEST(BoundaryTest, DomainBoundaryEnumValues)
{
  EXPECT_EQ(hmap::DomainBoundary::BOUNDARY_LEFT, 0);
  EXPECT_EQ(hmap::DomainBoundary::BOUNDARY_RIGHT, 1);
  EXPECT_EQ(hmap::DomainBoundary::BOUNDARY_TOP, 2);
  EXPECT_EQ(hmap::DomainBoundary::BOUNDARY_BOTTOM, 3);
}

TEST(BoundaryTest, PickBoundaryCellFlatArray)
{
  hmap::Array z(glm::ivec2(16, 16), 5.0f);

  glm::ivec2 cell_b = hmap::pick_boundary_cell(z, hmap::DomainBoundary::BOUNDARY_BOTTOM, 42);
  EXPECT_EQ(cell_b.y, 0);
  EXPECT_GE(cell_b.x, 0);
  EXPECT_LT(cell_b.x, 16);

  glm::ivec2 cell_t = hmap::pick_boundary_cell(z, hmap::DomainBoundary::BOUNDARY_TOP, 42);
  EXPECT_EQ(cell_t.y, 15);
  EXPECT_GE(cell_t.x, 0);
  EXPECT_LT(cell_t.x, 16);

  glm::ivec2 cell_l = hmap::pick_boundary_cell(z, hmap::DomainBoundary::BOUNDARY_LEFT, 42);
  EXPECT_EQ(cell_l.x, 0);
  EXPECT_GE(cell_l.y, 0);
  EXPECT_LT(cell_l.y, 16);

  glm::ivec2 cell_r = hmap::pick_boundary_cell(z, hmap::DomainBoundary::BOUNDARY_RIGHT, 42);
  EXPECT_EQ(cell_r.x, 15);
  EXPECT_GE(cell_r.y, 0);
  EXPECT_LT(cell_r.y, 16);
}

TEST(BoundaryTest, GenerateBufferedArrayPivotCentered)
{
  hmap::Array arr(glm::ivec2(5, 5));
  for (int j = 0; j < 5; ++j)
    for (int i = 0; i < 5; ++i)
      arr(i, j) = float(i + j * 10);

  glm::ivec4 buffers(2, 2, 2, 2);
  hmap::Array buffered = hmap::generate_buffered_array(arr, buffers, false);

  EXPECT_EQ(buffered.shape.x, 9);
  EXPECT_EQ(buffered.shape.y, 9);

  // Interior is in [2, 6] (West pivot is 2, East pivot is 6)
  // For West buffer:
  // cell 1 (distance 1 to pivot 2) mirrors cell 3:
  EXPECT_FLOAT_EQ(buffered(1, 3), buffered(3, 3));
  // cell 0 (distance 2 to pivot 2) mirrors cell 4:
  EXPECT_FLOAT_EQ(buffered(0, 3), buffered(4, 3));

  // For East buffer:
  // cell 7 (distance 1 to pivot 6) mirrors cell 5:
  EXPECT_FLOAT_EQ(buffered(7, 3), buffered(5, 3));
  // cell 8 (distance 2 to pivot 6) mirrors cell 4:
  EXPECT_FLOAT_EQ(buffered(8, 3), buffered(4, 3));
}

TEST(BoundaryTest, FalloffWithAndWithoutNoise)
{
  hmap::Array a1(glm::ivec2(16, 16), 1.f);
  hmap::falloff(a1, 1.f);

  hmap::Array noise(glm::ivec2(16, 16), 0.f);
  hmap::Array a2(glm::ivec2(16, 16), 1.f);
  hmap::falloff(a2, 1.f, hmap::DistanceFunction::EUCLIDIAN, &noise);

  for (int j = 0; j < 16; ++j)
    for (int i = 0; i < 16; ++i)
      EXPECT_NEAR(a1(i, j), a2(i, j), 1e-5f);
}
