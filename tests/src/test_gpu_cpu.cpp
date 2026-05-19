#include "highmap.hpp"
#include "highmap/dbg/assert.hpp"
#include "highmap/dbg/timer.hpp"

#include <gtest/gtest.h>

using namespace hmap;

Array helper_generate_array()
{
  const glm::ivec2 shape = {512, 512};
  const glm::vec2  kw = {4.f, 4.f};
  const int        seed = 0;

  Array z = noise_fbm(NoiseType::PERLIN, shape, kw, seed);
  return z;
}

TEST(GpuCpu, ClosingByReconstruction)
{
  Timer::Clear();

  const float tol = 1e0f;
  const int   ir = 64;

  Array z = helper_generate_array();

  {
    Timer::Start("CPU");
    Array zc = closing_by_reconstruction(z, ir);
    Timer::Stop("CPU");

    Timer::Start("GPU");
    Array zg = gpu::closing_by_reconstruction(z, ir);
    Timer::Stop("GPU");

    bool ret = assert_almost_equal(zc, zg, tol);
    EXPECT_EQ(ret, true);
  }

  {
    Timer::Start("CPU");
    Array zc = opening_by_reconstruction(z, ir);
    Timer::Stop("CPU");

    Timer::Start("GPU");
    Array zg = gpu::opening_by_reconstruction(z, ir);
    Timer::Stop("GPU");

    bool ret = assert_almost_equal(zc, zg, tol);
    EXPECT_EQ(ret, true);
  }

  auto map = Timer::DumpDurations();
  for (auto [k, v] : map)
    RecordProperty(k, v);
}

TEST(GpuCpu, MorphologicalOperators)
{
  Timer::Clear();

  const float tol = 1e0f;
  const int   ir = 32;

  Array z = helper_generate_array();

  std::vector<MorphologyOperation> ops = {
      MorphologyOperation::MO_BORDER,
      MorphologyOperation::MO_CLOSING,
      MorphologyOperation::MO_DILATION,
      MorphologyOperation::MO_EROSION,
      MorphologyOperation::MO_OPENING,
      MorphologyOperation::MO_GRADIENT,
      MorphologyOperation::MO_TOP_HAT,
      MorphologyOperation::MO_BLACK_HAT,
      MorphologyOperation::MO_LAPLACIAN,
      MorphologyOperation::MO_CLOSING_BY_RECONSTRUCTION,
      MorphologyOperation::MO_OPENING_BY_RECONSTRUCTION,
  };

  for (const auto op : ops)
  {
    Timer::Start("CPU");
    Array zc = morphological_operators(z, ir, op);
    Timer::Stop("CPU");

    Timer::Start("GPU");
    Array zg = gpu::morphological_operators(z, ir, op);
    Timer::Stop("GPU");

    bool ret = assert_almost_equal(zc, zg, tol);
    EXPECT_EQ(ret, true);
  }

  auto map = Timer::DumpDurations();
  for (auto [k, v] : map)
    RecordProperty(k, v);
}
