#include <gtest/gtest.h>

#include "highmap.hpp"
#include "highmap/dbg/assert.hpp"
#include "highmap/dbg/timer.hpp"

hmap::Array helper_generate_array()
{
  const glm::ivec2 shape = {512, 512};
  const glm::vec2  kw = {4.f, 4.f};
  const int        seed = 0;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  return z;
}

TEST(GpuCpu, ClosingByReconstruction)
{
  hmap::gpu::init_opencl();
  hmap::Timer::Clear();

  const float tol = 1e0f;
  const int   ir = 64;

  hmap::Array z = helper_generate_array();

  {
    hmap::Timer::Start("CPU");
    hmap::Array zc = hmap::closing_by_reconstruction(z, ir);
    hmap::Timer::Stop("CPU");

    hmap::Timer::Start("GPU");
    hmap::Array zg = hmap::gpu::closing_by_reconstruction(z, ir);
    hmap::Timer::Stop("GPU");

    bool ret = hmap::assert_almost_equal(zc, zg, tol);
    EXPECT_EQ(ret, true);
  }

  {
    hmap::Timer::Start("CPU");
    hmap::Array zc = hmap::opening_by_reconstruction(z, ir);
    hmap::Timer::Stop("CPU");

    hmap::Timer::Start("GPU");
    hmap::Array zg = hmap::gpu::opening_by_reconstruction(z, ir);
    hmap::Timer::Stop("GPU");

    bool ret = hmap::assert_almost_equal(zc, zg, tol);
    EXPECT_EQ(ret, true);
  }

  auto map = hmap::Timer::DumpDurations();
  for (auto [k, v] : map)
    RecordProperty(k, v);
}

TEST(GpuCpu, MorphologicalOperators)
{
  hmap::gpu::init_opencl();
  hmap::Timer::Clear();

  const float tol = 1e0f;
  const int   ir = 32;

  hmap::Array z = helper_generate_array();

  std::vector<hmap::MorphologyOperation> ops = {
      hmap::MorphologyOperation::MO_BORDER,
      hmap::MorphologyOperation::MO_CLOSING,
      hmap::MorphologyOperation::MO_DILATION,
      hmap::MorphologyOperation::MO_EROSION,
      hmap::MorphologyOperation::MO_OPENING,
      hmap::MorphologyOperation::MO_GRADIENT,
      hmap::MorphologyOperation::MO_TOP_HAT,
      hmap::MorphologyOperation::MO_BLACK_HAT,
      hmap::MorphologyOperation::MO_LAPLACIAN,
      hmap::MorphologyOperation::MO_CLOSING_BY_RECONSTRUCTION,
      hmap::MorphologyOperation::MO_OPENING_BY_RECONSTRUCTION,
  };

  for (const auto op : ops)
  {
    hmap::Timer::Start("CPU");
    hmap::Array zc = hmap::morphological_operators(z, ir, op);
    hmap::Timer::Stop("CPU");

    hmap::Timer::Start("GPU");
    hmap::Array zg = hmap::gpu::morphological_operators(z, ir, op);
    hmap::Timer::Stop("GPU");

    bool ret = hmap::assert_almost_equal(zc, zg, tol);
    EXPECT_EQ(ret, true);
  }

  auto map = hmap::Timer::DumpDurations();
  for (auto [k, v] : map)
    RecordProperty(k, v);
}
