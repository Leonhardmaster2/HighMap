#include "highmap.hpp"

#include <gtest/gtest.h>

// Simple interpolation on a two-point path.
TEST(PathSampleAt, InterpolatesPositionValueAndTangent)
{
  hmap::Path path(
      std::vector<hmap::Point>{{0.f, 0.f, 0.f}, {10.f, 0.f, 100.f}});

  glm::vec2 tangent;
  glm::vec3 sample = path.sample_at(0.5f, nullptr, &tangent);

  EXPECT_NEAR(sample.x, 5.f, 1e-5f);
  EXPECT_NEAR(sample.y, 0.f, 1e-5f);
  EXPECT_NEAR(sample.z, 50.f, 1e-5f);

  EXPECT_NEAR(tangent.x, 1.f, 1e-5f);
  EXPECT_NEAR(tangent.y, 0.f, 1e-5f);
}

// Sampling exactly at arc-length control points.
TEST(PathSampleAt, SamplesAtControlPoints)
{
  hmap::Path path(std::vector<hmap::Point>{{0.f, 0.f, 10.f},
                                           {1.f, 0.f, 20.f},
                                           {4.f, 0.f, 40.f}});

  std::vector<float> arc = path.get_arc_length();

  glm::vec3 s0 = path.sample_at(arc[0]);
  glm::vec3 s1 = path.sample_at(arc[1]);
  glm::vec3 s2 = path.sample_at(arc[2]);

  EXPECT_NEAR(s0.x, 0.f, 1e-5f);
  EXPECT_NEAR(s0.y, 0.f, 1e-5f);
  EXPECT_NEAR(s0.z, 10.f, 1e-5f);

  EXPECT_NEAR(s1.x, 1.f, 1e-5f);
  EXPECT_NEAR(s1.y, 0.f, 1e-5f);
  EXPECT_NEAR(s1.z, 20.f, 1e-5f);

  EXPECT_NEAR(s2.x, 4.f, 1e-5f);
  EXPECT_NEAR(s2.y, 0.f, 1e-5f);
  EXPECT_NEAR(s2.z, 40.f, 1e-5f);
}

// Verify that t is clamped into [0, 1].
TEST(PathSampleAt, ClampsParameter)
{
  hmap::Path path(
      std::vector<hmap::Point>{{0.f, 0.f, 0.f}, {10.f, 0.f, 100.f}});

  glm::vec3 before = path.sample_at(-1.f);
  glm::vec3 after = path.sample_at(2.f);

  EXPECT_NEAR(before.x, 0.f, 1e-5f);
  EXPECT_NEAR(before.y, 0.f, 1e-5f);
  EXPECT_NEAR(before.z, 0.f, 1e-5f);

  EXPECT_NEAR(after.x, 10.f, 1e-5f);
  EXPECT_NEAR(after.y, 0.f, 1e-5f);
  EXPECT_NEAR(after.z, 100.f, 1e-5f);
}

// Non-uniform path: verify interpolation follows arc length.
TEST(PathSampleAt, UsesArcLengthParameterization)
{
  hmap::Path path(std::vector<hmap::Point>{{0.f, 0.f, 0.f},
                                           {1.f, 0.f, 10.f},
                                           {4.f, 0.f, 40.f}});

  // Arc-length values are [0, 0.25, 1].
  glm::vec3 sample = path.sample_at(0.5f);

  // 0.5 lies in the second segment:
  // r = (0.5 - 0.25) / (1 - 0.25) = 1/3.
  EXPECT_NEAR(sample.x, 2.f, 1e-5f);
  EXPECT_NEAR(sample.y, 0.f, 1e-5f);
  EXPECT_NEAR(sample.z, 20.f, 1e-5f);
}

// Verify that a supplied arc vector is used.
TEST(PathSampleAt, UsesProvidedArcVector)
{
  hmap::Path path(std::vector<hmap::Point>{{0.f, 0.f, 0.f},
                                           {1.f, 0.f, 10.f},
                                           {2.f, 0.f, 20.f}});

  // Artificial parameterization.
  std::vector<float> arc{0.f, 0.5f, 1.f};

  glm::vec3 sample = path.sample_at(0.25f, &arc);

  EXPECT_NEAR(sample.x, 0.5f, 1e-5f);
  EXPECT_NEAR(sample.y, 0.f, 1e-5f);
  EXPECT_NEAR(sample.z, 5.f, 1e-5f);
}

// Degenerate segment: tangent should fall back to (1, 0).
TEST(PathSampleAt, DegenerateSegmentUsesDefaultTangent)
{
  hmap::Path path(std::vector<hmap::Point>{{1.f, 2.f, 3.f}, {1.f, 2.f, 5.f}});

  glm::vec2 tangent;
  path.sample_at(0.5f, nullptr, &tangent);

  EXPECT_NEAR(tangent.x, 1.f, 1e-5f);
  EXPECT_NEAR(tangent.y, 0.f, 1e-5f);
}
