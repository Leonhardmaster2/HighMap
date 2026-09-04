/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the LICENSE file. */

#include <algorithm>
#include <array>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "highmap.hpp"
#include "highmap/gpu/metal.hpp"
#include "highmap/opencl/gpu_opencl.hpp"
#include "opencl_test_utils.hpp"

namespace
{

using hmap::Array;

void fill_field(Array &array, float phase = 0.f)
{
  for (int j = 0; j < array.shape.y; ++j)
    for (int i = 0; i < array.shape.x; ++i)
      array(i, j) = 0.013f * float(i * i) + 0.071f * float(j) +
                    0.19f * std::sin(0.17f * float(i + j) + phase);
}

void fill_bounded_field(Array &array, float phase = 0.f)
{
  for (int j = 0; j < array.shape.y; ++j)
    for (int i = 0; i < array.shape.x; ++i)
      array(i, j) = 0.35f * std::sin(0.037f * float(i) + phase) +
                    0.25f * std::cos(0.061f * float(j) - phase) +
                    0.1f * std::sin(0.013f * float(i + j));
}

void expect_close(const Array &actual, const Array &expected, float tolerance)
{
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.vector.size(), expected.vector.size());
  for (std::size_t i = 0; i < actual.vector.size(); ++i)
  {
    ASSERT_TRUE(std::isfinite(actual.vector[i])) << "index " << i;
    ASSERT_NEAR(actual.vector[i], expected.vector[i], tolerance)
        << "index " << i;
  }
}

struct ErrorMetrics
{
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double rmse = 0.0;
};

ErrorMetrics error_metrics(const Array &actual, const Array &expected)
{
  ErrorMetrics result;
  double squared_sum = 0.0;
  for (std::size_t i = 0; i < actual.vector.size(); ++i)
  {
    const double error = std::fabs(double(actual.vector[i]) -
                                   double(expected.vector[i]));
    result.max_abs = std::max(result.max_abs, error);
    result.mean_abs += error;
    squared_sum += error * error;
  }
  const double count = static_cast<double>(actual.vector.size());
  result.mean_abs /= count;
  result.rmse = std::sqrt(squared_sum / count);
  return result;
}

void expect_sweep_result(const char       *operation,
                         int                resolution,
                         const Array       &actual,
                         const Array       &expected,
                         double             tolerance)
{
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.vector.size(), expected.vector.size());
  for (const float value : actual.vector) ASSERT_TRUE(std::isfinite(value));

  const ErrorMetrics metrics = error_metrics(actual, expected);
  std::cout << "Metal small sweep " << operation << " " << resolution << "x"
            << resolution << ": max_abs=" << metrics.max_abs
            << " mean_abs=" << metrics.mean_abs << " rmse=" << metrics.rmse
            << '\n';
  EXPECT_LE(metrics.max_abs, tolerance);
}

bool metal_available()
{
  return hmap::gpu::metal::is_available();
}

} // namespace

TEST(MetalPortability, UnavailableBackendContractIsDeterministic)
{
  if (hmap::gpu::metal::is_available())
    GTEST_SKIP() << "The native Metal backend is available";

  EXPECT_TRUE(hmap::gpu::metal::device_name().empty());
  const auto capabilities = hmap::gpu::metal::capabilities();
  EXPECT_TRUE(capabilities.device_name.empty());
  EXPECT_EQ(capabilities.thread_execution_width, 0u);
  EXPECT_EQ(capabilities.max_threads_per_threadgroup, 0u);
  EXPECT_EQ(hmap::gpu::metal::last_execution_stats().upload_count, 0u);

  hmap::gpu::metal::DeviceArray empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.shape(), glm::ivec2(0, 0));
  EXPECT_EQ(empty.size(), 0u);
  EXPECT_THROW(empty.to_array(), std::runtime_error);
  EXPECT_THROW({ hmap::gpu::metal::DeviceSession session; }, std::runtime_error);
}

TEST(MetalPortability, PublicGpuWrapperRemainsUsableWithoutMetal)
{
  if (hmap::gpu::metal::is_available())
    GTEST_SKIP() << "The native Metal backend is available";
  if (!hmap::gpu::init_opencl())
    GTEST_SKIP() << "No OpenCL device is available for the established fallback";

  const glm::ivec2 shape = {31, 17};
  Array input(shape);
  fill_field(input, 0.4f);
  Array other(shape);
  fill_field(other, -0.3f);

  const Array gradient = hmap::gpu::gradient_norm(input);
  expect_close(gradient, hmap::gradient_norm(input), 2e-5f);

  const Array actual = hmap::gpu::maximum_smooth(input, other, 0.2f);
  expect_close(actual, hmap::maximum_smooth(input, other, 0.2f), 2e-5f);
}

TEST(MetalHardening, SessionTransferStatisticsCountHostCopies)
{
  if (!metal_available())
    GTEST_SKIP() << "No usable Metal device or shader compiler is available";

  const glm::ivec2 shape = {37, 19};
  Array input(shape);
  fill_field(input);

  hmap::gpu::metal::DeviceSession session(
      hmap::gpu::metal::StorageMode::private_storage);
  auto uploaded = session.upload(
      input, hmap::gpu::metal::StorageMode::private_storage);
  const auto after_upload = session.stats();
  EXPECT_EQ(after_upload.upload_count, 1u);
  EXPECT_EQ(after_upload.upload_bytes, input.vector.size() * sizeof(float));
  EXPECT_EQ(after_upload.readback_count, 0u);
  EXPECT_EQ(after_upload.readback_bytes, 0u);

  // normalize reads only its two scalar range values before encoding the
  // pointwise pass. The terrain remains device-resident until download().
  auto normalized = session.normalize(std::move(uploaded));
  const auto after_normalize = session.stats();
  EXPECT_EQ(after_normalize.upload_count, 1u);
  EXPECT_EQ(after_normalize.readback_count, 1u);
  EXPECT_EQ(after_normalize.readback_bytes, 2u * sizeof(float));
  EXPECT_EQ(normalized.residency_state(),
            hmap::gpu::metal::ResidencyState::device_valid);

  const Array actual = session.download(normalized);
  const auto after_download = session.stats();
  EXPECT_EQ(after_download.upload_count, 1u);
  EXPECT_EQ(after_download.readback_count, 2u);
  EXPECT_EQ(after_download.readback_bytes,
            2u * sizeof(float) + actual.vector.size() * sizeof(float));
  Array expected = input;
  hmap::remap(expected, 0.f, 1.f);
  expect_close(actual, expected, 3e-5f);
}

TEST(MetalHardening, SynchronousWrapperTransferStatisticsMatchCopies)
{
  if (!metal_available())
    GTEST_SKIP() << "No usable Metal device or shader compiler is available";

  Array input(glm::ivec2{19, 13});
  fill_field(input, 0.6f);
  const Array actual = hmap::gpu::metal::gradient_norm(input);
  const auto stats = hmap::gpu::metal::last_execution_stats();

  EXPECT_EQ(stats.upload_count, 1u);
  EXPECT_EQ(stats.upload_bytes, input.vector.size() * sizeof(float));
  EXPECT_EQ(stats.readback_count, 1u);
  EXPECT_EQ(stats.readback_bytes, actual.vector.size() * sizeof(float));
  expect_close(actual, hmap::gradient_norm(input), 2e-5f);
}

TEST(MetalHardening, ConstantNormalizationMatchesHighMapRangeContract)
{
  if (!metal_available())
    GTEST_SKIP() << "No usable Metal device or shader compiler is available";

  const std::vector<glm::ivec2> shapes = {{1, 257}, {37, 19}, {255, 257}};
  const std::vector<float> values = {-3.f, 0.f, 4.f};
  for (const glm::ivec2 shape : shapes)
    for (const float value : values)
    {
      Array input(shape, value);
      hmap::gpu::metal::DeviceSession session;
      auto normalized = session.normalize(session.upload(input), -0.25f, 0.75f);
      const Array actual = session.download(normalized);
      ASSERT_EQ(actual.shape, shape);
      for (const float result : actual.vector)
      {
        ASSERT_TRUE(std::isfinite(result));
        EXPECT_FLOAT_EQ(result, -0.25f);
      }
    }
}

TEST(MetalHardening, ThermalRidgeMatchesSynchronousMetalAcrossShapes)
{
  if (!metal_available())
    GTEST_SKIP() << "No usable Metal device or shader compiler is available";

  for (const glm::ivec2 shape : {glm::ivec2{29, 13},
                                 glm::ivec2{63, 37},
                                 glm::ivec2{127, 65}})
  {
    Array source(shape);
    fill_field(source, 0.35f);
    Array talus(shape);
    fill_field(talus, -0.4f);
    for (float &value : talus.vector)
      value = 0.008f + 0.004f * std::fabs(value - talus.vector.front());

    for (const int iterations : {1, 3, 7})
    {
      Array expected = source;
      hmap::gpu::metal::thermal_ridge(expected, talus, iterations);

      hmap::gpu::metal::DeviceSession session;
      auto result = session.thermal_ridge(session.upload(source),
                                           session.upload(talus),
                                           iterations);
      result = session.extrapolate_borders(std::move(result));
      const Array actual = session.download(result);
      expect_close(actual, expected, 2e-5f);
    }
  }
}

TEST(MetalHardening, SmallResolutionCoreParitySweep)
{
  if (!metal_available())
    GTEST_SKIP() << "No usable Metal device or shader compiler is available";
  HMAP_SKIP_IF_NO_OPENCL();

  constexpr std::array<int, 6> resolutions = {32, 64, 128, 256, 512, 1024};
  for (const int resolution : resolutions)
  {
    const glm::ivec2 shape = {resolution, resolution};
    Array input(shape);
    fill_bounded_field(input, 0.2f);
    Array other(shape);
    fill_bounded_field(other, -0.7f);

    expect_sweep_result("gradient", resolution,
                        hmap::gpu::metal::gradient_norm(input),
                        hmap::gradient_norm(input),
                        2e-5);
    expect_sweep_result("maximum_smooth", resolution,
                        hmap::gpu::metal::maximum_smooth(input, other, 0.2f),
                        hmap::maximum_smooth(input, other, 0.2f),
                        2e-5);
    expect_sweep_result("morphological_gradient", resolution,
                        hmap::gpu::metal::morphological_gradient(input, 1),
                        hmap::gpu::morphological_gradient(input, 1),
                        2e-5);

    Array expected_normalized = input;
    hmap::remap(expected_normalized, -0.25f, 0.75f);
    hmap::gpu::metal::DeviceSession normalize_session;
    const Array actual_normalized = normalize_session.download(
        normalize_session.normalize(normalize_session.upload(input),
                                    -0.25f,
                                    0.75f));
    expect_sweep_result("normalize", resolution, actual_normalized,
                        expected_normalized, 4e-5);

    const Array talus(shape, 0.01f);
    Array expected_thermal = input;
    hmap::gpu::metal::thermal(expected_thermal, talus, 2);
    hmap::gpu::metal::DeviceSession thermal_session;
    const Array actual_thermal = thermal_session.download(
        thermal_session.thermal(thermal_session.upload(input),
                                thermal_session.upload(talus),
                                2));
    expect_sweep_result("thermal", resolution, actual_thermal,
                        expected_thermal, 2e-5);

    Array expected_ridge = input;
    hmap::gpu::metal::thermal_ridge(expected_ridge, talus, 2);
    hmap::gpu::metal::DeviceSession ridge_session;
    auto ridge_result = ridge_session.thermal_ridge(
        ridge_session.upload(input), ridge_session.upload(talus), 2);
    ridge_result = ridge_session.extrapolate_borders(std::move(ridge_result));
    const Array actual_ridge = ridge_session.download(ridge_result);
    expect_sweep_result("thermal_ridge", resolution, actual_ridge,
                        expected_ridge, 2e-5);
  }
}

TEST(MetalHardening, OddNonSquareReductionsCoverAllElements)
{
  if (!metal_available())
    GTEST_SKIP() << "No usable Metal device or shader compiler is available";

  for (const glm::ivec2 shape : {glm::ivec2{127, 63}, glm::ivec2{255, 257}})
  {
    Array input(shape);
    fill_field(input, 0.8f);
    Array expected = input;
    hmap::remap(expected, -0.5f, 0.75f);

    hmap::gpu::metal::DeviceSession session;
    auto normalized = session.normalize(session.upload(input), -0.5f, 0.75f);
    const Array actual = session.download(normalized);
    expect_close(actual, expected, 4e-5f);
  }
}

TEST(MetalHardening, ExplicitTransferPreservesNonFiniteValues)
{
  if (!metal_available())
    GTEST_SKIP() << "No usable Metal device or shader compiler is available";

  Array input(glm::ivec2{4, 3}, 0.f);
  input.vector[0] = std::numeric_limits<float>::quiet_NaN();
  input.vector[1] = std::numeric_limits<float>::infinity();
  input.vector[2] = -std::numeric_limits<float>::infinity();
  input.vector[7] = -1.25f;

  hmap::gpu::metal::DeviceSession session;
  auto device = session.upload(input);
  const Array actual = session.download(device);
  ASSERT_TRUE(std::isnan(actual.vector[0]));
  EXPECT_TRUE(std::isinf(actual.vector[1]));
  EXPECT_GT(actual.vector[1], 0.f);
  EXPECT_TRUE(std::isinf(actual.vector[2]));
  EXPECT_LT(actual.vector[2], 0.f);
  EXPECT_FLOAT_EQ(actual.vector[7], -1.25f);
}

TEST(MetalHardening, CompletedAdoptionSurvivesRepeatedSessionChaining)
{
  if (!metal_available())
    GTEST_SKIP() << "No usable Metal device or shader compiler is available";

  const glm::ivec2 shape = {23, 17};
  Array input(shape);
  fill_field(input, 1.1f);

  for (int iteration = 0; iteration < 64; ++iteration)
  {
    hmap::gpu::metal::DeviceArray completed;
    {
      hmap::gpu::metal::DeviceSession producer;
      auto uploaded = producer.upload(input);
      completed = producer.gradient_norm(std::move(uploaded));
      producer.finish();
      EXPECT_EQ(producer.stats().resident_bytes, 0u);
    }

    hmap::gpu::metal::DeviceSession consumer;
    auto adopted = consumer.adopt_completed(completed);
    auto chained = consumer.gradient_norm(std::move(adopted));
    const Array actual = consumer.download(chained);
    ASSERT_EQ(actual.shape, shape);
    for (const float value : actual.vector)
      ASSERT_TRUE(std::isfinite(value));
    EXPECT_EQ(consumer.stats().resident_bytes, 0u);
  }
}

TEST(MetalHardening, MovedAndInvalidArraysFailWithoutCorruptingSession)
{
  if (!metal_available())
    GTEST_SKIP() << "No usable Metal device or shader compiler is available";

  hmap::gpu::metal::DeviceSession session;
  hmap::gpu::metal::DeviceArray empty;
  EXPECT_THROW(session.gradient_norm(empty), std::invalid_argument);

  auto original = session.upload(Array(glm::ivec2{11, 7}, 1.f));
  auto moved = std::move(original);
  EXPECT_TRUE(original.empty());
  EXPECT_THROW(session.gradient_norm(std::move(original)), std::invalid_argument);
  auto result = session.gradient_norm(std::move(moved));
  EXPECT_EQ(session.download(result).shape, glm::ivec2(11, 7));

  hmap::gpu::metal::DeviceSession closed;
  closed.finish();
  EXPECT_THROW(closed.allocate({8, 8}), std::runtime_error);
}

TEST(MetalHardening, RepeatedResidentOperationsReuseSessionBuffers)
{
  if (!metal_available())
    GTEST_SKIP() << "No usable Metal device or shader compiler is available";

  hmap::gpu::metal::DeviceSession session;
  auto value = session.noise(hmap::NoiseType::PERLIN, {47, 31}, {4.f, 3.f}, 9u);
  for (int i = 0; i < 16; ++i)
    value = session.gradient_norm(std::move(value));

  const Array actual = session.download(value);
  const auto stats = session.stats();
  EXPECT_EQ(stats.upload_count, 0u);
  EXPECT_EQ(stats.readback_count, 1u);
  EXPECT_GT(stats.buffer_reuses, 0u);
  EXPECT_GT(stats.bytes_reused, 0u);
  EXPECT_GT(stats.peak_resident_bytes, actual.vector.size() * sizeof(float));
  for (const float element : actual.vector)
    EXPECT_TRUE(std::isfinite(element));
}

TEST(MetalHardening, IndependentSessionsCanRunConcurrently)
{
  if (!metal_available())
    GTEST_SKIP() << "No usable Metal device or shader compiler is available";

  const glm::ivec2 shape = {29, 17};
  const auto run = [shape](std::uint32_t seed) {
    hmap::gpu::metal::DeviceSession session;
    auto noise = session.noise(hmap::NoiseType::PERLIN,
                               shape,
                               {3.5f, 4.25f},
                               seed);
    auto gradient = session.gradient_norm(std::move(noise));
    return session.download(gradient);
  };

  auto first = std::async(std::launch::async, run, 101u);
  auto second = std::async(std::launch::async, run, 202u);
  const Array first_result = first.get();
  const Array second_result = second.get();
  const Array first_expected = hmap::gpu::metal::gradient_norm(
      hmap::gpu::metal::noise(hmap::NoiseType::PERLIN,
                              shape,
                              {3.5f, 4.25f},
                              101u));
  const Array second_expected = hmap::gpu::metal::gradient_norm(
      hmap::gpu::metal::noise(hmap::NoiseType::PERLIN,
                              shape,
                              {3.5f, 4.25f},
                              202u));
  expect_close(first_result, first_expected, 2e-5f);
  expect_close(second_result, second_expected, 2e-5f);
}
