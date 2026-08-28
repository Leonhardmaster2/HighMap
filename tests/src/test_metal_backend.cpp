/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the LICENSE file. */

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>
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
                   std::uint32_t seed,
                   const Array  *p_noise_x = nullptr,
                   const Array  *p_noise_y = nullptr,
                   glm::vec4     bbox = {0.f, 1.f, 0.f, 1.f},
                   glm::ivec2    period = {0, 0})
{
  Array out(shape);
  std::vector<float> dummy(1, 0.f);
  clwrapper::Run run("noise");
  run.bind_buffer<float>("array", out.vector);
  run.bind_buffer<float>("noise_x",
                         p_noise_x ? const_cast<std::vector<float> &>(
                                         p_noise_x->vector)
                                   : dummy);
  run.bind_buffer<float>("noise_y",
                         p_noise_y ? const_cast<std::vector<float> &>(
                                         p_noise_y->vector)
                                   : dummy);
  run.bind_arguments(shape.x,
                     shape.y,
                     static_cast<int>(noise_type),
                     kw.x,
                     kw.y,
                     seed,
                     p_noise_x ? 1 : 0,
                     p_noise_y ? 1 : 0,
                     period.x,
                     period.y,
                     bbox);
  run.write_buffer("array");
  if (p_noise_x) run.write_buffer("noise_x");
  if (p_noise_y) run.write_buffer("noise_y");
  run.execute({shape.x, shape.y});
  run.read_buffer("array");
  return out;
}

Array opencl_advection(const Array &z,
                       const Array &field,
                       const Array &dx,
                       const Array &dy,
                       float        advection_length,
                       float        value_persistence,
                       const Array *p_mask = nullptr)
{
  Array out(z.shape);
  Array mask(z.shape, 1.f);
  clwrapper::Run run("advection_warp");
  run.bind_imagef("z", z.vector, z.shape.x, z.shape.y);
  run.bind_imagef("field", field.vector, z.shape.x, z.shape.y);
  run.bind_imagef("dx", dx.vector, z.shape.x, z.shape.y);
  run.bind_imagef("dy", dy.vector, z.shape.x, z.shape.y);
  run.bind_imagef("mask",
                  p_mask ? p_mask->vector : mask.vector,
                  z.shape.x,
                  z.shape.y);
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

Array thermal_ridge_reference(Array z, const Array &talus, int iterations)
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
        float sum = 0.f;
        float slope_max = 0.f;
        for (int k = 0; k < 8; ++k)
        {
          const float dz = (value - z(i + di[k], j + dj[k])) / distance[k];
          if (dz > 0.f) sum += dz;
          slope_max = std::max(slope_max, std::fabs(dz));
        }

        const float t = talus(i, j);
        float amp = slope_max > 0.f ? std::clamp(1.f - t / slope_max, 0.f, 1.f)
                                    : 0.f;
        amp = amp * amp * (3.f - 2.f * amp);
        next(i, j) = value + 0.25f * (t - 0.5f * sum) * amp;
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
  const auto capabilities = hmap::gpu::metal::capabilities();
  EXPECT_FALSE(capabilities.device_name.empty());
  EXPECT_GT(capabilities.recommended_max_working_set_size, 0u);
  EXPECT_GT(capabilities.thread_execution_width, 0u);
  EXPECT_GT(capabilities.max_threads_per_threadgroup, 0u);
}

TEST_F(MetalBackend, DeviceArraySharedUploadDownloadAndMove)
{
  const glm::ivec2 shape = {17, 11};
  Array input(shape);
  fill_field(input);

  hmap::gpu::metal::DeviceSession session;
  auto device = session.upload(input);
  EXPECT_FALSE(device.empty());
  EXPECT_EQ(device.shape(), shape);
  EXPECT_EQ(device.size(), input.vector.size());
  EXPECT_EQ(device.storage_mode(), hmap::gpu::metal::StorageMode::shared);
  EXPECT_EQ(device.residency_state(),
            hmap::gpu::metal::ResidencyState::both_valid);

  auto moved = std::move(device);
  EXPECT_TRUE(device.empty());
  expect_finite_and_close(session.download(moved), input, 0.f);

  const auto stats = session.stats();
  EXPECT_EQ(stats.command_buffers, 1u);
  EXPECT_EQ(stats.synchronization_count, 0u);
  EXPECT_EQ(stats.readback_bytes, input.vector.size() * sizeof(float));
}

TEST_F(MetalBackend, DeviceArrayPrivateUploadDownloadUsesExplicitStaging)
{
  const glm::ivec2 shape = {19, 7};
  Array input(shape);
  fill_field(input);

  hmap::gpu::metal::DeviceSession session(
      hmap::gpu::metal::StorageMode::private_storage);
  auto device = session.upload(
      input, hmap::gpu::metal::StorageMode::private_storage);
  const auto actual = session.download(device);
  expect_finite_and_close(actual, input, 0.f);

  const auto stats = session.stats();
  EXPECT_EQ(stats.command_buffers, 1u);
  EXPECT_EQ(stats.synchronization_count, 1u);
  EXPECT_GE(stats.blit_encoders, 2u);
  EXPECT_EQ(stats.readback_bytes, input.vector.size() * sizeof(float));
}

TEST_F(MetalBackend, DeviceArrayCanOutliveSession)
{
  const glm::ivec2 shape = {13, 9};
  Array input(shape);
  fill_field(input);

  hmap::gpu::metal::DeviceArray device;
  {
    hmap::gpu::metal::DeviceSession session;
    device = session.upload(input);
    device.set_debug_name("phase3-lifetime");
    EXPECT_EQ(device.debug_name(), "phase3-lifetime");
  }

  EXPECT_EQ(device.residency_state(),
            hmap::gpu::metal::ResidencyState::both_valid);
  expect_finite_and_close(device.to_array(), input, 0.f);
}

TEST_F(MetalBackend, DeviceArrayCanBeAdoptedByCompletedSession)
{
  const glm::ivec2 shape = {17, 13};
  Array input(shape);
  fill_field(input);

  hmap::gpu::metal::DeviceArray completed;
  {
    hmap::gpu::metal::DeviceSession producer;
    completed = producer.gradient_norm(producer.upload(input));
    producer.finish();
  }

  hmap::gpu::metal::DeviceSession consumer;
  auto adopted = consumer.adopt_completed(completed);
  const Array actual = consumer.download(adopted);
  const Array expected = hmap::gpu::metal::gradient_norm(input);
  expect_finite_and_close(actual, expected, 2e-5f);
}

TEST_F(MetalBackend, DeviceArrayRejectsMixedSessionsAndShapeMismatch)
{
  Array first_input({11, 7}, 1.f);
  Array second_input({13, 7}, 2.f);
  hmap::gpu::metal::DeviceSession first_session;
  hmap::gpu::metal::DeviceSession second_session;
  auto first = first_session.upload(first_input);
  auto second = second_session.upload(second_input);
  auto wrong_shape = first_session.upload(second_input);

  EXPECT_THROW(first_session.gradient_norm(second), std::invalid_argument);
  EXPECT_THROW(first_session.maximum_smooth(first, second, 0.2f),
               std::invalid_argument);
  EXPECT_THROW(first_session.maximum_smooth(first, wrong_shape, 0.2f),
               std::invalid_argument);
}

TEST_F(MetalBackend, DeviceArrayFinishClosesSessionOnce)
{
  Array input({15, 9}, 0.5f);
  hmap::gpu::metal::DeviceSession session;
  auto device = session.upload(input);
  session.finish();
  const auto finished = session.stats();
  EXPECT_EQ(finished.synchronization_count, 0u);
  EXPECT_THROW(session.gradient_norm(std::move(device)), std::runtime_error);
}

TEST_F(MetalBackend, DeviceSessionCanSplitCommandBuffersWithoutWaiting)
{
  const glm::ivec2 shape = {21, 13};
  hmap::gpu::metal::DeviceSession session;
  auto noise = session.noise(
      hmap::NoiseType::PERLIN, shape, {4.f, 4.f}, 42u);
  session.submit();
  auto gradient = session.gradient_norm(std::move(noise));
  const Array actual = session.download(gradient);
  const Array expected = hmap::gpu::metal::gradient_norm(
      hmap::gpu::metal::noise(hmap::NoiseType::PERLIN,
                              shape,
                              {4.f, 4.f},
                              42u));
  expect_finite_and_close(actual, expected, 1e-5f);

  const auto stats = session.stats();
  EXPECT_EQ(stats.command_buffers, 2u);
  EXPECT_EQ(stats.synchronization_count, 1u);
}

TEST_F(MetalBackend, DeviceArrayPrivateDownloadIsIdempotent)
{
  Array input({13, 11}, 0.25f);
  hmap::gpu::metal::DeviceSession session(
      hmap::gpu::metal::StorageMode::private_storage);
  auto device = session.upload(
      input, hmap::gpu::metal::StorageMode::private_storage);
  const Array first = session.download(device);
  const auto after_first = session.stats();
  const Array second = session.download(device);
  const auto after_second = session.stats();

  expect_finite_and_close(first, input, 0.f);
  expect_finite_and_close(second, input, 0.f);
  EXPECT_EQ(after_first.synchronization_count, 1u);
  EXPECT_EQ(after_second.synchronization_count, 1u);
  EXPECT_EQ(after_second.readback_bytes,
            2u * input.vector.size() * sizeof(float));
  EXPECT_EQ(device.residency_state(),
            hmap::gpu::metal::ResidencyState::both_valid);
}

TEST_F(MetalBackend, DeviceArrayComposedPointwiseMatchesSynchronousMetal)
{
  const glm::ivec2 shape = {33, 19};
  Array source(shape);
  fill_field(source);

  const Array synchronous_noise = hmap::gpu::metal::noise(
      hmap::NoiseType::PERLIN, shape, {4.f, 4.f}, 42u);
  const Array synchronous_gradient =
      hmap::gpu::metal::gradient_norm(synchronous_noise);
  const Array expected = hmap::gpu::metal::maximum_smooth(
      synchronous_gradient, synchronous_noise, 0.2f);

  hmap::gpu::metal::DeviceSession session;
  auto device_noise = session.noise(
      hmap::NoiseType::PERLIN, shape, {4.f, 4.f}, 42u);
  auto device_other_noise = session.noise(
      hmap::NoiseType::PERLIN, shape, {4.f, 4.f}, 42u);
  auto device_gradient = session.gradient_norm(std::move(device_noise));
  auto device_result = session.maximum_smooth(
      std::move(device_gradient), device_other_noise, 0.2f);
  const Array actual = session.download(device_result);
  expect_finite_and_close(actual, expected, 1e-5f);

  const auto stats = session.stats();
  EXPECT_EQ(stats.command_buffers, 1u);
  EXPECT_EQ(stats.synchronization_count, 1u);
  EXPECT_EQ(stats.readback_bytes, actual.vector.size() * sizeof(float));
  EXPECT_EQ(stats.upload_bytes, 0u);
  EXPECT_GE(stats.buffer_reuses, 1u);
  EXPECT_GT(stats.bytes_reused, 0u);
}

TEST_F(MetalBackend, DeviceArrayTextureAdvectionMatchesBufferPath)
{
  const glm::ivec2 shape = {23, 15};
  Array z(shape);
  Array field(shape);
  Array dx(shape, 0.35f);
  Array dy(shape, -0.2f);
  fill_field(z);
  fill_field(field);

  const Array expected = hmap::gpu::metal::advection_warp(
      z, field, dx, dy, 0.35f, 0.8f, nullptr);
  hmap::gpu::metal::DeviceSession session;
  auto device_z = session.upload(z);
  auto device_field = session.upload(field);
  auto device_dx = session.upload(dx);
  auto device_dy = session.upload(dy);
  auto device_result = session.advection_warp_texture(
      std::move(device_z), device_field, device_dx, device_dy, 0.35f, 0.8f);
  const Array actual = session.download(device_result);
  expect_finite_and_close(actual, expected, 2e-3f);

  const auto stats = session.stats();
  EXPECT_EQ(stats.command_buffers, 1u);
  EXPECT_EQ(stats.synchronization_count, 1u);
  EXPECT_GE(stats.texture_allocations, 6u);
  EXPECT_GE(stats.blit_encoders, 6u);
}

TEST_F(MetalBackend, DeviceArrayThermalMatchesSynchronousMetal)
{
  const glm::ivec2 shape = {29, 13};
  Array source(shape);
  fill_field(source);
  const Array talus(shape, 0.01f);

  Array expected = source;
  hmap::gpu::metal::thermal(expected, talus, 5);

  hmap::gpu::metal::DeviceSession session(
      hmap::gpu::metal::StorageMode::private_storage);
  auto device_source = session.upload(
      source, hmap::gpu::metal::StorageMode::private_storage);
  auto device_talus = session.upload(
      talus, hmap::gpu::metal::StorageMode::private_storage);
  auto device_result =
      session.thermal(std::move(device_source), device_talus, 5);
  const Array actual = session.download(device_result);
  expect_finite_and_close(actual, expected, 2e-5f);

  const auto stats = session.stats();
  EXPECT_EQ(stats.command_buffers, 1u);
  EXPECT_EQ(stats.synchronization_count, 1u);
  EXPECT_EQ(stats.encoders, 5u);
}

TEST_F(MetalBackend, DeviceArrayBorderExtrapolationMatchesArray)
{
  const glm::ivec2 shape = {29, 13};
  Array source(shape);
  fill_field(source);
  Array expected = source;
  hmap::extrapolate_borders(expected);

  hmap::gpu::metal::DeviceSession session;
  auto device_source = session.upload(source);
  auto device_result = session.extrapolate_borders(std::move(device_source));
  const Array actual = session.download(device_result);
  expect_finite_and_close(actual, expected, 1e-6f);
}

TEST_F(MetalBackend, DeviceArrayLinearThermalBlendMatchesReference)
{
  const glm::ivec2 shape = {29, 13};
  Array source(shape);
  fill_field(source);
  const Array talus(shape, 0.01f);
  const int standard_iterations = 4;
  const int ridge_iterations = 3;
  const float weight1 = 0.8f;
  const float weight2 = 1.1f;

  const Array expected_thermal = thermal_ridge_reference(
      thermal_reference(source, talus, standard_iterations),
      talus,
      ridge_iterations);
  Array expected(expected_thermal.shape);
  for (std::size_t i = 0; i < expected.vector.size(); ++i)
    expected.vector[i] = weight1 * source.vector[i] +
                         weight2 * expected_thermal.vector[i];

  hmap::gpu::metal::DeviceSession session;
  auto device_source = session.upload(source);
  auto device_talus = session.upload(talus);
  auto device_thermal = session.thermal(
      std::move(device_source), device_talus, standard_iterations);
  device_thermal = session.thermal_ridge(
      std::move(device_thermal), device_talus, ridge_iterations);
  auto device_result = session.linear_combine(
      session.upload(source), device_thermal, weight1, weight2);
  const Array actual = session.download(device_result);

  expect_finite_and_close(actual, expected, 2e-5f);
  const auto stats = session.stats();
  EXPECT_EQ(stats.command_buffers, 1u);
  EXPECT_EQ(stats.synchronization_count, 1u);
  EXPECT_EQ(stats.encoders,
            static_cast<std::uint64_t>(standard_iterations + ridge_iterations + 1));
}

TEST_F(MetalBackend, DeviceArrayHydraulicMatchesSynchronousMetal)
{
  const glm::ivec2 shape = {17, 11};
  Array source(shape);
  fill_field(source);
  Array expected = source;
  Array expected_water;
  hmap::gpu::metal::hydraulic_vpipes(expected,
                                     0.025f,
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
                                     &expected_water,
                                     nullptr,
                                     nullptr,
                                     nullptr);

  hmap::gpu::metal::DeviceSession session(
      hmap::gpu::metal::StorageMode::private_storage);
  auto device_source = session.upload(
      source, hmap::gpu::metal::StorageMode::private_storage);
  hmap::gpu::metal::DeviceArray actual_water;
  auto device_result = session.hydraulic_vpipes(std::move(device_source),
                                                0.025f,
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
                                                &actual_water);
  const Array actual = session.download(device_result);
  const Array actual_water_host = session.download(actual_water);
  expect_finite_and_close(actual, expected, 2e-5f);
  expect_finite_and_close(actual_water_host, expected_water, 2e-5f);

  const auto stats = session.stats();
  EXPECT_EQ(stats.command_buffers, 1u);
  EXPECT_EQ(stats.synchronization_count, 1u);
  EXPECT_EQ(stats.readback_bytes,
            (actual.vector.size() + actual_water_host.vector.size()) *
                sizeof(float));
  EXPECT_GT(stats.peak_resident_bytes, actual.vector.size() * sizeof(float));
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

TEST_F(MetalBackend, ResidentNoiseFbmAndNormalizationMatchOpenCL)
{
  if (!opencl_available())
    GTEST_SKIP() << "No OpenCL device is available for parity comparison";

  const glm::ivec2 shape = {37, 29};
  const glm::vec2 kw = {5.f, 3.f};
  const std::uint32_t seed = 1234u;
  Array expected = hmap::gpu::noise_fbm(hmap::NoiseType::SIMPLEX2,
                                        shape,
                                        kw,
                                        seed,
                                        8,
                                        0.7f,
                                        0.5f,
                                        2.f);
  hmap::remap(expected, 0.f, 1.f);

  hmap::gpu::metal::DeviceSession session;
  auto device = session.noise_fbm(hmap::NoiseType::SIMPLEX2,
                                  shape,
                                  kw,
                                  seed,
                                  8,
                                  0.7f,
                                  0.5f,
                                  2.f);
  auto normalized = session.normalize(std::move(device));
  const Array actual = session.download(normalized);
  expect_finite_and_close(actual, expected, 2e-4f);
  EXPECT_EQ(session.stats().upload_bytes, 0u);
}

TEST_F(MetalBackend, ResidentGaborWaveFbmMatchesOpenCL)
{
  if (!opencl_available())
    GTEST_SKIP() << "No OpenCL device is available for parity comparison";

  const glm::ivec2 shape = {37, 29};
  const glm::vec2 kw = {3.5f, 5.25f};
  const std::uint32_t seed = 2718u;
  const float angle_degrees = 21.2f;
  Array ctrl(shape);
  Array noise_x(shape);
  Array noise_y(shape);
  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
    {
      const float phase = 0.17f * float(i) + 0.11f * float(j);
      ctrl(i, j) = 0.35f + 0.25f * std::sin(phase);
      noise_x(i, j) = 0.015f * std::sin(0.23f * float(i + j));
      noise_y(i, j) = 0.02f * std::cos(0.19f * float(i - j));
    }

  Array angle_delta(shape);
  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
      angle_delta(i, j) = 0.03f * std::sin(0.13f * float(i)) *
                          std::cos(0.09f * float(j));

  Array angle_field(shape, angle_degrees);
  angle_field += angle_delta * (180.f / float(M_PI));

  const Array expected = hmap::gpu::gabor_wave_fbm(shape,
                                                    kw,
                                                    seed,
                                                    angle_field,
                                                    0.65f,
                                                    6,
                                                    0.72f,
                                                    0.53f,
                                                    1.9f,
                                                    &ctrl,
                                                    &noise_x,
                                                    &noise_y,
                                                    {-0.25f, 1.25f, 0.1f, 0.9f});
  const Array actual = hmap::gpu::metal::gabor_wave_fbm(
      shape,
      kw,
      seed,
      angle_degrees,
      0.65f,
      6,
      0.72f,
      0.53f,
      1.9f,
      &ctrl,
      &noise_x,
      &noise_y,
      &angle_delta,
      {-0.25f, 1.25f, 0.1f, 0.9f});

  expect_finite_and_close(actual, expected, 3e-4f);
}

TEST_F(MetalBackend, ResidentSpectralEqualizerMatchesOpenCL)
{
  if (!opencl_available())
    GTEST_SKIP() << "No OpenCL device is available for parity comparison";

  const glm::ivec2 shape = {47, 31};
  Array source(shape);
  fill_field(source);
  const std::vector<float> weights = {0.8f, 1.1f, 0.6f, 1.3f, 0.9f, 1.0f};
  const Array expected =
      hmap::gpu::spectral_equalizer(source, weights, 3, 11);

  hmap::gpu::metal::DeviceSession session;
  auto device_source = session.upload(source);
  auto device_result = session.spectral_equalizer(std::move(device_source),
                                                  weights,
                                                  3,
                                                  11);
  const Array actual = session.download(device_result);
  expect_finite_and_close(actual, expected, 2e-4f);
  EXPECT_EQ(session.stats().readback_bytes,
            actual.vector.size() * sizeof(float));
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

TEST_F(MetalBackend, BoundaryAndFlatGradientStress)
{
  const glm::ivec2 shape = {2, 23};
  Array input(shape, -3.5f);
  const Array expected = hmap::gradient_norm(input);
  const Array actual = hmap::gpu::metal::gradient_norm(input);
  expect_finite_and_close(actual, expected, 1e-6f);

  Array non_square(glm::ivec2{29, 7});
  fill_field(non_square);
  const Array expected_non_square = hmap::gradient_norm(non_square);
  const Array actual_non_square =
      hmap::gpu::metal::gradient_norm(non_square);
  expect_finite_and_close(actual_non_square, expected_non_square, 1e-5f);
}

TEST_F(MetalBackend, NoiseOptionalInputsPeriodsAndSeeds)
{
  if (!opencl_available())
    GTEST_SKIP() << "No OpenCL device is available for parity comparison";

  const glm::ivec2 shape = {31, 11};
  Array noise_x(shape);
  Array noise_y(shape);
  for (size_t k = 0; k < noise_x.vector.size(); ++k)
  {
    noise_x.vector[k] = 0.15f * std::sin(float(k));
    noise_y.vector[k] = -0.1f * std::cos(0.5f * float(k));
  }

  for (const std::uint32_t seed : {1u, 42u, 0xffffffffu})
  {
    const auto actual = hmap::gpu::metal::noise(
        hmap::NoiseType::VALUE_LINEAR,
        shape,
        {3.2f, 2.7f},
        seed,
        &noise_x,
        &noise_y,
        {-0.25f, 1.25f, -0.5f, 1.5f},
        {5, 3});
    const auto expected = opencl_noise(hmap::NoiseType::VALUE_LINEAR,
                                       shape,
                                       {3.2f, 2.7f},
                                       seed,
                                       &noise_x,
                                       &noise_y,
                                       {-0.25f, 1.25f, -0.5f, 1.5f},
                                       {5, 3});
    expect_finite_and_close(actual, expected, 2e-5f);
  }
}

TEST_F(MetalBackend, AdvectionOptionalMaskAndBoundaryStress)
{
  if (!opencl_available())
    GTEST_SKIP() << "No OpenCL device is available for parity comparison";

  const glm::ivec2 shape = {11, 7};
  Array z(shape);
  Array field(shape);
  Array dx(shape);
  Array dy(shape);
  Array mask(shape);
  fill_field(z);
  fill_field(field);
  for (size_t k = 0; k < dx.vector.size(); ++k)
  {
    dx.vector[k] = (k % 3 == 0) ? -0.4f : 0.25f;
    dy.vector[k] = (k % 4 == 0) ? 0.3f : -0.15f;
    mask.vector[k] = (k % 2 == 0) ? 0.f : 1.f;
  }

  const auto actual = hmap::gpu::metal::advection_warp(
      z, field, dx, dy, 1.5f, 0.65f, &mask);
  const auto expected =
      opencl_advection(z, field, dx, dy, 1.5f, 0.65f, &mask);
  expect_finite_and_close(actual, expected, 2e-3f);
}

TEST_F(MetalBackend, ThermalIterationCountStress)
{
  const glm::ivec2 shape = {33, 19};
  Array source(shape);
  fill_field(source);
  const Array talus(shape, 0.01f);

  for (const int iterations : {0, 1, 10})
  {
    Array actual = source;
    const Array expected = thermal_reference(source, talus, iterations);
    hmap::gpu::metal::thermal(actual, talus, iterations);
    expect_finite_and_close(actual, expected, 2e-5f);
  }
}

TEST_F(MetalBackend, HydraulicOptionalOutputsAndNonSquareStress)
{
  const glm::ivec2 shape = {19, 13};
  Array terrain(shape);
  Array rain(shape);
  fill_field(terrain);
  for (size_t k = 0; k < rain.vector.size(); ++k)
    rain.vector[k] = 0.5f + 0.5f * std::fabs(std::sin(float(k)));

  Array water_depth;
  Array sediment;
  Array velocity_u;
  Array velocity_v;
  hmap::gpu::metal::hydraulic_vpipes(terrain,
                                     0.025f,
                                     true,
                                     0.15f,
                                     10,
                                     0.25f,
                                     0.4f,
                                     0.002f,
                                     0.02f,
                                     1.2f,
                                     8.f,
                                     false,
                                     0.f,
                                     &rain,
                                     &water_depth,
                                     &sediment,
                                     &velocity_u,
                                     &velocity_v);

  const auto execution = hmap::gpu::metal::last_execution_stats();
  EXPECT_EQ(execution.command_buffers, 1u);
  EXPECT_EQ(execution.synchronization_count, 1u);
  EXPECT_GE(execution.encoders, 50u);
  EXPECT_GT(execution.readback_bytes, 0u);

  for (const Array *result : {&terrain,
                              &water_depth,
                              &sediment,
                              &velocity_u,
                              &velocity_v})
  {
    ASSERT_EQ(result->shape.x, shape.x);
    ASSERT_EQ(result->shape.y, shape.y);
    for (const float value : result->vector) EXPECT_TRUE(std::isfinite(value));
  }

  const double expected_water_volume =
      0.025 * std::accumulate(rain.vector.begin(), rain.vector.end(), 0.f);
  const double actual_water_volume =
      std::accumulate(water_depth.vector.begin(),
                      water_depth.vector.end(),
                      0.0);
  EXPECT_NEAR(actual_water_volume, expected_water_volume, 5e-4);
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
