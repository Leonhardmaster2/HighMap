/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the LICENSE file. */

#include <gtest/gtest.h>

#include "highmap.hpp"
#include "highmap/gpu/metal.hpp"

namespace
{

hmap::Array make_field(glm::ivec2 shape, float seed_offset)
{
  hmap::Array a(shape);
  for (int j = 0; j < shape.y; ++j)
    for (int i = 0; i < shape.x; ++i)
      a(i, j) = 0.03f * float(i * i) + 0.07f * float(j) +
                0.2f * std::sin(0.4f * float(i + j) + seed_offset);
  return a;
}

double max_abs_diff(const hmap::Array &a, const hmap::Array &b)
{
  double m = 0.0;
  for (size_t i = 0; i < a.vector.size(); ++i)
    m = std::max(m, std::abs(double(a.vector[i]) - double(b.vector[i])));
  return m;
}

} // namespace

class MetalPhase8 : public ::testing::Test
{
protected:
  void SetUp() override
  {
    if (!hmap::gpu::metal::is_available())
      GTEST_SKIP() << "No usable Metal device";
  }
};

TEST_F(MetalPhase8, SpectralTerrainGraphKeepsResidencyWithOneReadback)
{
  const glm::ivec2 shape = {64, 64};
  hmap::gpu::metal::DeviceSession session;
  auto noise = session.noise_fbm(hmap::NoiseType::PERLIN,
                                 shape,
                                 {4.f, 4.f},
                                 42u,
                                 8,
                                 0.7f,
                                 0.5f,
                                 2.f);
  noise = session.normalize(std::move(noise), 0.f, 1.f);
  const std::vector<float> weights = {0.f, 0.f, 1.f, 1.f, 1.f, 1.f};
  const int ir_min = std::max(1, int(0.05f * shape.x));
  const int ir_max = std::max(1, int(0.25f * shape.x));
  auto spectral = session.spectral_equalizer(std::move(noise), weights, ir_min, ir_max);
  const hmap::Array talus(shape, 0.01f);
  auto talus_d = session.upload(talus);
  auto thermal = session.thermal(std::move(spectral), talus_d, 4);
  thermal = session.extrapolate_borders(std::move(thermal));
  auto source = session.noise_fbm(hmap::NoiseType::PERLIN,
                                  shape,
                                  {4.f, 4.f},
                                  42u,
                                  8,
                                  0.7f,
                                  0.5f,
                                  2.f);
  source = session.normalize(std::move(source), 0.f, 1.f);
  // Re-create spectral branch for consistent comparison: use thermal output
  // blended with original noise via linear_combine
  auto out = session.linear_combine(std::move(source), thermal, 1.f, 1.f);
  const hmap::Array host = session.download(out);
  const auto stats = session.stats();
  EXPECT_EQ(stats.upload_bytes, talus.vector.size() * sizeof(float));
  // Two normalizes each do a scalar reduction read (8 bytes each) plus the
  // terminal terrain readback; allow that small overhead.
  EXPECT_GE(stats.readback_bytes, host.vector.size() * sizeof(float));
  EXPECT_LE(stats.readback_bytes,
            host.vector.size() * sizeof(float) + 32u);
  // Two normalize reductions each introduce a command-buffer split plus the
  // final finish, so expect 3 command buffers / synchronizations.
  EXPECT_EQ(stats.synchronization_count, 3u);
  EXPECT_EQ(stats.command_buffers, 3u);
  EXPECT_GE(stats.encoders, 10u);
  EXPECT_GE(stats.peak_resident_bytes, host.vector.size() * sizeof(float));
  for (float v : host.vector) EXPECT_TRUE(std::isfinite(v));
}

TEST_F(MetalPhase8, GeneratedNoiseChainHasZeroUploadsAndOneReadback)
{
  const glm::ivec2 shape = {64, 64};
  hmap::gpu::metal::DeviceSession session;
  auto n = session.noise(hmap::NoiseType::PERLIN, shape, {4.f, 4.f}, 42u);
  auto g = session.gradient_norm(std::move(n));
  auto n2 = session.noise(hmap::NoiseType::PERLIN, shape, {4.f, 4.f}, 42u);
  auto r = session.maximum_smooth(std::move(g), n2, 0.2f);
  const hmap::Array host = session.download(r);
  const auto stats = session.stats();
  EXPECT_EQ(stats.upload_bytes, 0u);
  EXPECT_EQ(stats.readback_bytes, host.vector.size() * sizeof(float));
  EXPECT_EQ(stats.command_buffers, 1u);
  EXPECT_EQ(stats.synchronization_count, 1u);
  EXPECT_GE(stats.buffer_reuses, 1u);
}

TEST_F(MetalPhase8, HybridBoundaryUploadsOnlyAtGenuineCpuMetalTransition)
{
  const glm::ivec2 shape = {64, 64};
  hmap::Array host_input = make_field(shape, 0.f);
  hmap::gpu::metal::DeviceSession session;
  auto d = session.upload(host_input);
  auto s = session.spectral_equalizer(std::move(d), {1.f, 1.f, 1.f, 1.f, 1.f, 1.f}, 2, 8);
  auto h = session.download(s);
  const auto stats = session.stats();
  EXPECT_EQ(stats.upload_bytes, host_input.vector.size() * sizeof(float));
  EXPECT_EQ(stats.readback_bytes, host_input.vector.size() * sizeof(float));
  EXPECT_EQ(stats.command_buffers, 1u);
  EXPECT_EQ(stats.synchronization_count, 1u);
  // Verify numerical parity with synchronous wrapper
  const hmap::Array ref = hmap::gpu::metal::spectral_equalizer(host_input, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f}, 2, 8);
  EXPECT_LE(max_abs_diff(h, ref), 2e-4);
}

TEST_F(MetalPhase8, UnsupportedNoiseTypeFallsBackWithException)
{
  hmap::gpu::metal::DeviceSession session;
  EXPECT_THROW(session.noise(static_cast<hmap::NoiseType>(999), {16, 16}, {1.f, 1.f}, 0u),
               std::invalid_argument);
  EXPECT_THROW(session.noise_fbm(static_cast<hmap::NoiseType>(999), {16, 16}, {1.f, 1.f}, 0u, 1, 0.5f, 0.5f, 2.f),
               std::invalid_argument);
}

TEST_F(MetalPhase8, RepeatedExecutionRequiresNewSessionAfterFinish)
{
  const glm::ivec2 shape = {32, 32};
  hmap::gpu::metal::DeviceSession s1;
  auto a = s1.noise(hmap::NoiseType::PERLIN, shape, {2.f, 2.f}, 1u);
  (void)s1.download(a);
  EXPECT_THROW(s1.noise(hmap::NoiseType::PERLIN, shape, {2.f, 2.f}, 2u), std::runtime_error);
  hmap::gpu::metal::DeviceSession s2;
  auto b = s2.noise(hmap::NoiseType::PERLIN, shape, {2.f, 2.f}, 2u);
  const hmap::Array hb = s2.download(b);
  EXPECT_EQ(hb.shape, shape);
  for (float v : hb.vector) EXPECT_TRUE(std::isfinite(v));
}

TEST_F(MetalPhase8, AdoptCompletedInvalidatesOnShapeMismatchViaMiss)
{
  const glm::ivec2 s1 = {16, 16};
  const glm::ivec2 s2 = {32, 32};
  hmap::gpu::metal::DeviceSession prod;
  auto d1 = prod.noise(hmap::NoiseType::PERLIN, s1, {2.f, 2.f}, 1u);
  prod.finish();
  hmap::gpu::metal::DeviceSession cons;
  auto adopted = cons.adopt_completed(d1);
  EXPECT_FALSE(adopted.empty());
  EXPECT_EQ(adopted.shape(), s1);
  // Adopted resource has shape s1; using it where s2 is expected should throw
  auto d2 = cons.noise(hmap::NoiseType::PERLIN, s2, {2.f, 2.f}, 2u);
  EXPECT_THROW(cons.maximum_smooth(std::move(adopted), d2, 0.2f), std::invalid_argument);
}

TEST_F(MetalPhase8, ThermalRidgePlusNormalizeReportsTwoCommandBuffers)
{
  const glm::ivec2 shape = {64, 64};
  hmap::Array src = make_field(shape, 1.f);
  hmap::Array talus(shape, 0.01f);
  hmap::gpu::metal::DeviceSession session;
  auto d = session.upload(src);
  auto talus_d = session.upload(talus);
  auto t = session.thermal(std::move(d), talus_d, 2);
  t = session.extrapolate_borders(std::move(t));
  // normalize triggers internal reduction sync -> 2 command buffers total
  auto n = session.normalize(std::move(t), 0.f, 1.f);
  (void)session.download(n);
  const auto stats = session.stats();
  EXPECT_EQ(stats.command_buffers, 2u);
  EXPECT_EQ(stats.synchronization_count, 2u);
}
