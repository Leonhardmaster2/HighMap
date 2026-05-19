#include <random>

#include "highmap/math/chebyshev.hpp"

#include <gtest/gtest.h>

using namespace hmap;

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

namespace
{

float reference_chebyshev(int n, float x)
{
  return std::cos(n * std::acos(x));
}

float reference_series(const std::vector<float> &c, float x)
{
  float result = 0.0;

  for (size_t k = 0; k < c.size(); ++k)
    result += c[k] * reference_chebyshev(static_cast<int>(k), x);

  return result;
}

} // namespace

// ------------------------------------------------------------
// Tests
// ------------------------------------------------------------

TEST(ChebyshevEvaluator, EmptyPolynomial)
{
  ChebyshevEvaluator eval(std::vector<float>{});

  EXPECT_EQ(eval.evaluate(0.0), 0.0);
  EXPECT_EQ(eval.evaluate(1.0), 0.0);
  EXPECT_EQ(eval.evaluate(-1.0), 0.0);
}

TEST(ChebyshevEvaluator, ConstantPolynomial)
{
  ChebyshevEvaluator eval({5.0});

  EXPECT_EQ(eval.evaluate(-1.0), 5.0);
  EXPECT_EQ(eval.evaluate(0.0), 5.0);
  EXPECT_EQ(eval.evaluate(1.0), 5.0);
}

TEST(ChebyshevEvaluator, FirstOrderPolynomial)
{
  // f(x) = 2*T1(x) = 2x
  ChebyshevEvaluator eval({0.0, 2.0});

  EXPECT_NEAR(eval.evaluate(-1.0), -2.0, 1e-12);
  EXPECT_NEAR(eval.evaluate(0.0), 0.0, 1e-12);
  EXPECT_NEAR(eval.evaluate(0.5), 1.0, 1e-12);
  EXPECT_NEAR(eval.evaluate(1.0), 2.0, 1e-12);
}

TEST(ChebyshevEvaluator, SecondOrderPolynomial)
{
  // T2(x) = 2x² - 1
  ChebyshevEvaluator eval({0.0, 0.0, 1.0});

  EXPECT_NEAR(eval.evaluate(-1.0), 1.0, 1e-12);
  EXPECT_NEAR(eval.evaluate(0.0), -1.0, 1e-12);
  EXPECT_NEAR(eval.evaluate(0.5), -0.5, 1e-12);
  EXPECT_NEAR(eval.evaluate(1.0), 1.0, 1e-12);
}

TEST(ChebyshevEvaluator, MixedPolynomial)
{
  // f(x) = 1*T0 + 2*T1 + 3*T2
  //
  // T0 = 1
  // T1 = x
  // T2 = 2x² - 1
  //
  // f(x) = 1 + 2x + 3(2x² - 1)
  //      = 6x² + 2x - 2

  ChebyshevEvaluator eval({1.0, 2.0, 3.0});

  EXPECT_NEAR(eval.evaluate(-1.0), 2.0, 1e-12);
  EXPECT_NEAR(eval.evaluate(0.0), -2.0, 1e-12);
  EXPECT_NEAR(eval.evaluate(0.5), 0.5, 1e-12);
  EXPECT_NEAR(eval.evaluate(1.0), 6.0, 1e-12);
}

TEST(ChebyshevEvaluator, CompareAgainstReferenceImplementation)
{
  constexpr int degree = 20;

  std::vector<float> coefficients(degree + 1);

  std::mt19937 rng(42);

  std::uniform_real_distribution<float> coeff_dist(-10.0, 10.0);
  std::uniform_real_distribution<float> x_dist(-1.0, 1.0);

  for (float &c : coefficients)
  {
    c = coeff_dist(rng);
  }

  ChebyshevEvaluator eval(coefficients);

  for (int i = 0; i < 1000; ++i)
  {
    const float x = x_dist(rng);
    const float expected = reference_series(coefficients, x);
    const float actual = eval.evaluate(x);

    EXPECT_NEAR(actual, expected, 1e-4f);
  }
}

TEST(ChebyshevEvaluator, BatchEvaluation)
{
  ChebyshevEvaluator eval({1.0, 2.0, 3.0});

  std::vector<float> input = {-1.0, -0.5, 0.0, 0.5, 1.0};
  std::vector<float> output(input.size());

  eval.evaluate_batch(input.begin(), input.end(), output.begin());

  for (size_t i = 0; i < input.size(); ++i)
  {
    const float expected = reference_series(eval.coefficients(), input[i]);
    EXPECT_NEAR(output[i], expected, 1e-4f);
  }
}

TEST(ChebyshevEvaluator, LargeNumberOfEvaluations)
{
  constexpr int iterations = 1'000'000;

  ChebyshevEvaluator eval({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

  volatile float accumulator = 0.0;

  for (int i = 0; i < iterations; ++i)
  {
    const float x = -1.0 + 2.0 * static_cast<float>(i) /
                               static_cast<float>(iterations);

    accumulator += eval.evaluate(x);
  }

  EXPECT_TRUE(std::isfinite(accumulator));
}
