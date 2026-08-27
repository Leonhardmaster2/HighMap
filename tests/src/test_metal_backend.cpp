/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the LICENSE file. */

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <utility>

#include <gtest/gtest.h>

#include "cl_wrapper/run.hpp"

#include "highmap.hpp"
#include "highmap/opencl/gpu_opencl.hpp"

namespace
{

using hmap::Array;

void fill_field(Array &array)
{
  for (int j = 0; j < array.shape.y; ++j)
    for (int i = 0; i < array.shape.x; ++i)
      array(i, j) = 0.03f * float(i * i) + 0.07f * float(j) +
                    0.2f * std::sin(0.4f * float(i + j));
}

Array opencl_gradient_norm(const Array &input)
{
  Array out(input.shape);
  clwrapper::Run run("gradient_norm");
  run.bind_buffer<float>("array",
                         const_cast<std::vector<float> &>(input.vector));
  run.bind_buffer<float>("dm", out.vector);
  run.bind_arguments(input.shape.x, input.shape.y);
  run.write_buffer("array");
  run.execute({input.shape.x, input.shape.y});
  run.read_buffer("dm");
  return out;
}

Array opencl_noise(hmap::NoiseType noise_type,
                   glm::ivec2    shape,
                   glm::vec2     kw,
                   std::uint32_t seed)
{
  Array out(shape);
  std::vector<float> dummy(1, 0.f);
  clwrapper::Run run("noise");
  run.bind_buffer<float>("array", out.vector);
  run.bind_buffer<float>("noise_x", dummy);
  run.bind_buffer<float>("noise_y", dummy);
  run.bind_arguments(shape.x,
                     shape.y,
                     static_cast<int>(noise_type),
                     kw.x,
                     kw.y,
                     seed,
                     0,
                     0,
                     0,
                     0,
                     glm::vec4(0.f, 1.f, 0.f, 1.f));
  run.write_buffer("array");
  run.execute({shape.x, shape.y});
  run.read_buffer("array");
  return out;
}

Array opencl_advection(const Array &z,
                       const Array &field,
                       const Array &dx,
                       const Array &dy,
                       float        advection_length,
                       float        value_persistence)
{
  Array out(z.shape);
  Array mask(z.shape, 1.f);
  clwrapper::Run run("advection_warp");
  run.bind_imagef("z", z.vector, z.shape.x, z.shape.y);
  run.bind_imagef("field", field.vector, z.shape.x, z.shape.y);
  run.bind_imagef("dx", dx.vector, z.shape.x, z.shape.y);
  run.bind_imagef("dy", dy.vector, z.shape.x, z.shape.y);
  run.bind_imagef("mask", mask.vector, z.shape.x, z.shape.y);
  run.bind_imagef("out", out.vector, z.shape.x, z.shape.y, true);
  run.bind_arguments(z.shape.x,
                     z.shape.y,
                     advection_length,
                     value_persistence);
  run.execute({z.shape.x, z.shape.y});
  run.read_imagef("out");
  return out;
}

Array thermal_reference(Array z, const Array &talus, int iterations)
{
  for (int iteration = 0; iteration < iterations; ++iteration)
  {
    Array next(z.shape);
    for (int j = 0; j < z.shape.y; ++j)
      for (int i = 0; i < z.shape.x; ++i)
      {
        if (i == 0 || i == z.shape.x - 1 || j == 0 || j == z.shape.y - 1)
        {
          next(i, j) = z(i, j);
          continue;
        }

        const int di[8] = {-1, 0, 0, 1, -1, -1, 1, 1};
        const int dj[8] = {0, 1, -1, 0, -1, 1, -1, 1};
        const float distance[8] = {
            1.f, 1.f, 1.f, 1.f, 1.414f, 1.414f, 1.414f, 1.414f};
        const float value = z(i, j);
        float amount = 0.f;
        for (int k = 0; k < 8; ++k)
        {
          const float other = z(i + di[k], j + dj[k]);
          const float difference = distance[k] * talus(i, j);
          if (value > other)
          {
            if (value - other > difference)
              amount -= 0.2f * ((value - other) - difference) / distance[k];
          }
          else if (other - value > difference)
            amount += 0.2f * ((other - value) - difference) / distance[k];
        }
        next(i, j) = value + amount;
      }
    z = std::move(next);
  }
  return z;
}

void expect_finite_and_close(const Array &actual,
                             const Array &expected,
                             float        tolerance)
{
  ASSERT_EQ(actual.shape.x, expected.shape.x);
  ASSERT_EQ(actual.shape.y, expected.shape.y);
  ASSERT_EQ(actual.vector.size(), expected.vector.size());

  double sum_abs = 0.0;
  double sum_squared = 0.0;
  float max_abs = 0.f;
  for (size_t k = 0; k < actual.vector.size(); ++k)
  {
    ASSERT_TRUE(std::isfinite(actual.vector[k])) << "index " << k;
    const float absolute_error =
        std::fabs(actual.vector[k] - expected.vector[k]);
    max_abs = std::max(max_abs, absolute_error);
    sum_abs += absolute_error;
    sum_squared += double(absolute_error) * double(absolute_error);
    EXPECT_LE(absolute_error, tolerance)
        << "index " << k;
  }

  const double count = static_cast<double>(actual.vector.size());
  ::testing::Test::RecordProperty("max_abs_error", max_abs);
  ::testing::Test::RecordProperty("mean_abs_error", sum_abs / count);
  ::testing::Test::RecordProperty("rmse", std::sqrt(sum_squared / count));
}

bool opencl_available()
{
  static const bool available = hmap::gpu::init_opencl();
  return available;
}

} // namespace

class MetalBackend : public ::testing::Test
{
protected:
  void SetUp() override
  {
    if (!hmap::gpu::metal::is_available())
      GTEST_SKIP() << "No usable Metal device or shader compiler is available";
  }
};

TEST_F(MetalBackend, ReportsUsableDevice)
{
  EXPECT_FALSE(hmap::gpu::metal::device_name().empty());
}

TEST_F(MetalBackend, GradientNormMatchesCpu)
{

  Array input(glm::ivec2(17, 11));
  fill_field(input);

  const Array expected = hmap::gradient_norm(input);
  const Array actual = hmap::gpu::metal::gradient_norm(input);
  expect_finite_and_close(actual, expected, 1e-5f);

  const Array routed = hmap::gpu::gradient_norm(input);
  expect_finite_and_close(routed, actual, 1e-5f);
}

TEST_F(MetalBackend, GradientNormMatchesOpenCL)
{
  if (!opencl_available())
    GTEST_SKIP() << "No OpenCL device is available for parity comparison";

  Array input(glm::ivec2(17, 11));
  fill_field(input);
  const Array actual = hmap::gpu::metal::gradient_norm(input);
  const Array opencl = opencl_gradient_norm(input);
  expect_finite_and_close(actual, opencl, 1e-5f);
}

TEST_F(MetalBackend, SmoothExtremaMatchesCpu)
{
  const glm::ivec2 shape = {17, 13};
  Array first(shape);
  Array second(shape);
  fill_field(first);
  fill_field(second);
  for (size_t k = 0; k < second.vector.size(); ++k)
    second.vector[k] = -0.4f * second.vector[k] + 0.1f;

  const float smoothing = 0.2f;
  const Array expected_max = hmap::maximum_smooth(first, second, smoothing);
  const Array expected_min = hmap::minimum_smooth(first, second, smoothing);
  const Array actual_max =
      hmap::gpu::metal::maximum_smooth(first, second, smoothing);
  const Array actual_min =
      hmap::gpu::metal::minimum_smooth(first, second, smoothing);
  expect_finite_and_close(actual_max, expected_max, 1e-5f);
  expect_finite_and_close(actual_min, expected_min, 1e-5f);
}

TEST_F(MetalBackend, SupportedNoiseIsDeterministicAndRouted)
{
  const glm::ivec2 shape = {19, 13};
  const glm::vec2 kw = {4.f, 3.f};
  const std::uint32_t seed = 1234u;
  const hmap::NoiseType noise_types[] = {
      hmap::NoiseType::PERLIN,
      hmap::NoiseType::PERLIN_BILLOW,
      hmap::NoiseType::PERLIN_HALF,
      hmap::NoiseType::SIMPLEX2,
      hmap::NoiseType::VALUE,
      hmap::NoiseType::VALUE_LINEAR};

  for (const hmap::NoiseType noise_type : noise_types)
  {
    const Array first = hmap::gpu::metal::noise(
        noise_type, shape, kw, seed, nullptr, nullptr, {0.f, 1.f, 0.f, 1.f});
    const Array second = hmap::gpu::metal::noise(
        noise_type, shape, kw, seed, nullptr, nullptr, {0.f, 1.f, 0.f, 1.f});
    const Array routed = hmap::gpu::noise(
        noise_type, shape, kw, seed, nullptr, nullptr, {0.f, 1.f, 0.f, 1.f});

    expect_finite_and_close(second, first, 1e-6f);
    expect_finite_and_close(routed, first, 1e-6f);
  }
}

TEST_F(MetalBackend, SupportedNoiseMatchesOpenCL)
{
  if (!opencl_available())
    GTEST_SKIP() << "No OpenCL device is available for parity comparison";

  const glm::ivec2 shape = {19, 13};
  const glm::vec2 kw = {4.f, 3.f};
  const std::uint32_t seed = 1234u;
  const hmap::NoiseType noise_types[] = {
      hmap::NoiseType::PERLIN,
      hmap::NoiseType::PERLIN_BILLOW,
      hmap::NoiseType::PERLIN_HALF,
      hmap::NoiseType::SIMPLEX2,
      hmap::NoiseType::VALUE,
      hmap::NoiseType::VALUE_LINEAR};

  for (const hmap::NoiseType noise_type : noise_types)
  {
    const Array actual = hmap::gpu::metal::noise(
        noise_type, shape, kw, seed, nullptr, nullptr, {0.f, 1.f, 0.f, 1.f});
    const Array opencl = opencl_noise(noise_type, shape, kw, seed);
    expect_finite_and_close(actual, opencl, 2e-5f);
  }
}

TEST_F(MetalBackend, AdvectionMatchesOpenCL)
{
  if (!opencl_available())
    GTEST_SKIP() << "No OpenCL device is available for parity comparison";

  const glm::ivec2 shape = {23, 15};
  Array z(shape);
  Array field(shape);
  Array dx(shape, 0.35f);
  Array dy(shape, -0.2f);
  fill_field(field);

  const Array actual = hmap::gpu::metal::advection_warp(
      z, field, dx, dy, 0.35f, 0.8f, nullptr);
  const Array opencl =
      opencl_advection(z, field, dx, dy, 0.35f, 0.8f);
  // The two runtimes take slightly different floating-point paths at the
  // mirrored boundary; the observed worst-case delta is below 1.4e-3.
  expect_finite_and_close(actual, opencl, 2e-3f);

  const Array routed = hmap::gpu::advection_warp(
      z, field, dx, dy, 0.35f, 0.8f, nullptr);
  expect_finite_and_close(routed, actual, 1e-5f);
}

TEST_F(MetalBackend, ThermalMatchesDeterministicReference)
{
  const glm::ivec2 shape = {21, 17};
  Array actual(shape);
  fill_field(actual);
  const Array expected = thermal_reference(actual, Array(shape, 0.01f), 4);
  const Array talus(shape, 0.01f);

  hmap::gpu::metal::thermal(actual, talus, 4);
  expect_finite_and_close(actual, expected, 1e-5f);
}

TEST_F(MetalBackend, HydraulicFlatFieldPreservesWaterAndTerrain)
{
  const glm::ivec2 shape = {17, 13};
  Array terrain(shape, 2.f);
  Array expected = terrain;
  Array water_depth;
  Array sediment;
  Array velocity_u;
  Array velocity_v;

  hmap::gpu::metal::hydraulic_vpipes(terrain,
                                     0.25f,
                                     true,
                                     0.1f,
                                     2,
                                     0.5f,
                                     0.5f,
                                     0.001f,
                                     0.01f,
                                     1.f,
                                     10.f,
                                     true,
                                     0.01f,
                                     nullptr,
                                     &water_depth,
                                     &sediment,
                                     &velocity_u,
                                     &velocity_v);

  expect_finite_and_close(terrain, expected, 1e-5f);
  EXPECT_EQ(water_depth.shape.x, shape.x);
  EXPECT_EQ(water_depth.shape.y, shape.y);
  EXPECT_EQ(sediment.shape.x, shape.x);
  EXPECT_EQ(sediment.shape.y, shape.y);
  EXPECT_EQ(velocity_u.shape.x, shape.x);
  EXPECT_EQ(velocity_u.shape.y, shape.y);
  EXPECT_EQ(velocity_v.shape.x, shape.x);
  EXPECT_EQ(velocity_v.shape.y, shape.y);
  for (size_t k = 0; k < water_depth.vector.size(); ++k)
  {
    EXPECT_NEAR(water_depth.vector[k], 0.25f, 2e-3f) << "index " << k;
    EXPECT_NEAR(sediment.vector[k], 0.f, 1e-5f) << "index " << k;
    EXPECT_NEAR(velocity_u.vector[k], 0.f, 1e-5f) << "index " << k;
    EXPECT_NEAR(velocity_v.vector[k], 0.f, 1e-5f) << "index " << k;
  }
}
