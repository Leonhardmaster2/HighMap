#include "highmap/dbg/assert.hpp"
#include "highmap/gradient.hpp"
#include "highmap/selector.hpp"

#include <gtest/gtest.h>

using namespace hmap;

TEST(SelectAngle, WrapAcrossZeroNorth)
{
  // create planes with known slopes (aspects)
  // aspect is downhill direction = atan2(-dy, -dx)
  auto make_plane = [](float alpha_deg)
  {
    Array z(glm::ivec2(5, 5));
    float alpha_rad = alpha_deg * float(M_PI) / 180.f;
    for (int j = 0; j < z.shape.y; ++j)
    {
      for (int i = 0; i < z.shape.x; ++i)
      {
        z(i, j) = -(std::cos(alpha_rad) * float(i) +
                    std::sin(alpha_rad) * float(j));
      }
    }
    return z;
  };

  // terrain with aspect at 5° (near 0°)
  Array z_5deg = make_plane(5.f);

  // select angle at 350° +/- 20° (should cover [330°, 370°] which includes 5°)
  Array sel = select_angle(z_5deg, 350.f, 20.f);

  // distance from 350° to 5° is 15°, which is inside tolerance 20°
  // smoothstep falloff at d/sigma = 15/20 = 0.75
  float expected_val = 1.f -
                       (3.f * 0.75f * 0.75f - 2.f * 0.75f * 0.75f * 0.75f);
  EXPECT_NEAR(sel(2, 2), expected_val, 1e-4);

  // select angle at 10° +/- 20° on aspect 355°
  Array z_355deg = make_plane(355.f);
  Array sel2 = select_angle(z_355deg, 10.f, 20.f);
  EXPECT_NEAR(sel2(2, 2), expected_val, 1e-4);
}

TEST(SelectAngle, BoundaryEquivalence0And360)
{
  Array z(glm::ivec2(8, 8));
  for (int j = 0; j < z.shape.y; ++j)
    for (int i = 0; i < z.shape.x; ++i)
      z(i, j) = std::sin(float(i)) * std::cos(float(j));

  Array sel_0 = select_angle(z, 0.f, 30.f);
  Array sel_360 = select_angle(z, 360.f, 30.f);
  Array sel_neg360 = select_angle(z, -360.f, 30.f);
  Array sel_720 = select_angle(z, 720.f, 30.f);

  EXPECT_TRUE(assert_almost_equal(sel_0, sel_360, 1e-5f));
  EXPECT_TRUE(assert_almost_equal(sel_0, sel_neg360, 1e-5f));
  EXPECT_TRUE(assert_almost_equal(sel_0, sel_720, 1e-5f));
}

TEST(SelectAngle, ExactMatchAndOutOfTolerance)
{
  auto make_plane = [](float alpha_deg)
  {
    Array z(glm::ivec2(5, 5));
    float alpha_rad = alpha_deg * float(M_PI) / 180.f;
    for (int j = 0; j < z.shape.y; ++j)
      for (int i = 0; i < z.shape.x; ++i)
        z(i, j) = -(std::cos(alpha_rad) * float(i) +
                    std::sin(alpha_rad) * float(j));
    return z;
  };

  Array z_90deg = make_plane(90.f);

  // exact match -> should be 1.0
  Array sel_exact = select_angle(z_90deg, 90.f, 30.f);
  EXPECT_NEAR(sel_exact(2, 2), 1.f, 1e-4);

  // completely outside tolerance -> should be 0.0
  Array sel_outside = select_angle(z_90deg, 150.f, 30.f);
  EXPECT_NEAR(sel_outside(2, 2), 0.f, 1e-4);

  // zero or negative sigma -> should be 0.0
  Array sel_zero_sigma = select_angle(z_90deg, 90.f, 0.f);
  EXPECT_NEAR(sel_zero_sigma(2, 2), 0.f, 1e-4);
}

TEST(SelectAngle, FlatTerrainExclusion)
{
  // perfectly flat array
  Array z_flat(glm::ivec2(6, 6), 5.f);

  // should return all 0s even when target angle is 0°
  Array sel = select_angle(z_flat, 0.f, 30.f);
  for (int j = 0; j < sel.shape.y; ++j)
    for (int i = 0; i < sel.shape.x; ++i)
      EXPECT_EQ(sel(i, j), 0.f);
}

TEST(SelectAngle, ValidationEmpty)
{
  Array empty;
  Array sel = select_angle(empty, 0.f, 10.f);
  EXPECT_EQ(sel.size(), 0);
}
