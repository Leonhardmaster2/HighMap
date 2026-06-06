#include "highmap.hpp"

#include <gtest/gtest.h>

using namespace hmap;

// Coincident x used to reach gsl_spline_init() and abort() the whole process
// (gsl: interp.c: "x values must be strictly increasing"). They must now be
// merged so the interpolator still builds and evaluates.
TEST(Interpolator1D, CoincidentXDoesNotCrash)
{
  std::vector<float> x = {0.f, 0.f, 0.5f, 1.f, 1.f};
  std::vector<float> y = {0.f, 0.f, 0.5f, 1.f, 1.f};

  EXPECT_NO_THROW({
    Interpolator1D itp(x, y, InterpolationMethod1D::LINEAR);
    EXPECT_NEAR(itp(0.f), 0.f, 1e-5f);
    EXPECT_NEAR(itp(0.5f), 0.5f, 1e-5f);
    EXPECT_NEAR(itp(1.f), 1.f, 1e-5f);
  });
}

// After merging coincident x there must still be at least two distinct points.
TEST(Interpolator1D, AllCoincidentXThrows)
{
  std::vector<float> x = {0.5f, 0.5f, 0.5f};
  std::vector<float> y = {0.f, 1.f, 2.f};

  EXPECT_THROW(Interpolator1D(x, y, InterpolationMethod1D::LINEAR),
               std::invalid_argument);
}

// Out-of-order (decreasing) x is unrepairable and must throw, not abort.
TEST(Interpolator1D, DecreasingXThrows)
{
  std::vector<float> x = {0.f, 1.f, 0.5f};
  std::vector<float> y = {0.f, 1.f, 0.5f};

  EXPECT_THROW(Interpolator1D(x, y, InterpolationMethod1D::LINEAR),
               std::invalid_argument);
}

// Existing preconditions remain enforced.
TEST(Interpolator1D, MismatchedOrTooFewPointsThrows)
{
  EXPECT_THROW(Interpolator1D(std::vector<float>{0.f, 1.f},
                              std::vector<float>{0.f},
                              InterpolationMethod1D::LINEAR),
               std::invalid_argument);

  EXPECT_THROW(Interpolator1D(std::vector<float>{0.f},
                              std::vector<float>{0.f},
                              InterpolationMethod1D::LINEAR),
               std::invalid_argument);
}
