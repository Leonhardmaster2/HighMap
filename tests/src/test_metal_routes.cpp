/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the LICENSE file. */

// Parity coverage for public hmap::gpu wrappers that route to Metal after the
// upstream sync: local min/max (DISK / SQUARE / OCTAGON), the morphology
// operators built on them, and smooth_cpulse.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "highmap/internal/opencl_run.hpp"

#include "highmap.hpp"
#include "highmap/gpu/metal.hpp"
#include "highmap/opencl/gpu_opencl.hpp"

namespace
{

using hmap::Array;
using hmap::MinMaxKernel;

bool metal_available()
{
  return hmap::gpu::metal::is_available();
}

bool opencl_available()
{
  static const bool available = hmap::gpu::init_opencl();
  return available;
}

void fill_field(Array &array)
{
  for (int j = 0; j < array.shape.y; ++j)
    for (int i = 0; i < array.shape.x; ++i)
      array(i, j) = 0.03f * float(i * i % 17) + 0.07f * float(j) +
                    0.4f * std::sin(0.37f * float(i) + 0.21f * float(j * j));
}

void expect_close(const Array &actual, const Array &expected, float tolerance)
{
  ASSERT_EQ(actual.shape, expected.shape);
  float max_error = 0.f;
  for (size_t k = 0; k < actual.vector.size(); ++k)
  {
    ASSERT_TRUE(std::isfinite(actual.vector[k])) << "index " << k;
    max_error = std::max(max_error,
                         std::abs(actual.vector[k] - expected.vector[k]));
  }
  EXPECT_LE(max_error, tolerance);
}

float clamped(const Array &a, int i, int j)
{
  return a(std::clamp(i, 0, a.shape.x - 1), std::clamp(j, 0, a.shape.y - 1));
}

// One 1D pass along (sx, sy) with clamp-to-edge sampling (OpenCL image
// sampler semantics).
Array line_pass(const Array &in, int ir, int sx, int sy, bool is_max)
{
  Array out(in.shape);
  for (int j = 0; j < in.shape.y; ++j)
    for (int i = 0; i < in.shape.x; ++i)
    {
      float v = in(i, j);
      for (int k = -ir; k <= ir; ++k)
      {
        const float s = clamped(in, i + k * sx, j + k * sy);
        v = is_max ? std::max(v, s) : std::min(v, s);
      }
      out(i, j) = v;
    }
  return out;
}

// Brute-force references of the OpenCL local_{max,min}_{disk,square,octagon}.
Array reference_extrema(const Array &in, int ir, MinMaxKernel kt, bool is_max)
{
  switch (kt)
  {
  case MinMaxKernel::DISK:
  {
    Array out(in.shape);
    for (int j = 0; j < in.shape.y; ++j)
      for (int i = 0; i < in.shape.x; ++i)
      {
        float v = in(i, j);
        for (int q = j - ir; q <= j + ir; ++q)
          for (int p = i - ir; p <= i + ir; ++p)
          {
            if (p < 0 || q < 0 || p >= in.shape.x || q >= in.shape.y) continue;
            if ((p - i) * (p - i) + (q - j) * (q - j) > ir * ir) continue;
            v = is_max ? std::max(v, in(p, q)) : std::min(v, in(p, q));
          }
        out(i, j) = v;
      }
    return out;
  }
  case MinMaxKernel::SQUARE:
    return line_pass(line_pass(in, ir, 1, 0, is_max), ir, 0, 1, is_max);
  case MinMaxKernel::OCTAGON:
  {
    const int b = static_cast<int>(
        std::round((std::sqrt(2.f) - 1.f) * static_cast<float>(ir)));
    const int a = ir - b;
    Array     out = line_pass(line_pass(in, a, 1, 0, is_max), a, 0, 1, is_max);
    if (b > 0)
      out = line_pass(line_pass(out, b, 1, 1, is_max), b, 1, -1, is_max);
    return out;
  }
  }
  return in;
}

// Direct OpenCL disk kernel, bypassing the routed wrapper.
Array opencl_disk(const Array &in, int ir, bool is_max)
{
  Array            out(in.shape);
  clwrapper::Run   run(is_max ? "local_max" : "local_min");
  std::vector<float> src = in.vector;
  run.bind_imagef("array", src, in.shape.x, in.shape.y);
  run.bind_imagef("out", out.vector, in.shape.x, in.shape.y, true);
  run.bind_arguments(in.shape.x, in.shape.y, ir);
  run.execute({in.shape.x, in.shape.y});
  run.read_imagef("out");
  return out;
}

const char *kernel_name(MinMaxKernel kt)
{
  switch (kt)
  {
  case MinMaxKernel::DISK: return "DISK";
  case MinMaxKernel::SQUARE: return "SQUARE";
  case MinMaxKernel::OCTAGON: return "OCTAGON";
  }
  return "?";
}

} // namespace

TEST(MetalRoutes, LocalExtremaMatchReference)
{
  if (!metal_available()) GTEST_SKIP() << "Metal backend is not available";

  Array input(glm::ivec2(41, 27));
  fill_field(input);

  for (const MinMaxKernel kt :
       {MinMaxKernel::DISK, MinMaxKernel::SQUARE, MinMaxKernel::OCTAGON})
    for (const int ir : {0, 1, 2, 5, 9, 30})
    {
      SCOPED_TRACE(std::string(kernel_name(kt)) + " ir=" + std::to_string(ir));
      expect_close(hmap::gpu::metal::local_max(input, ir, kt),
                   reference_extrema(input, ir, kt, true),
                   0.f);
      expect_close(hmap::gpu::metal::local_min(input, ir, kt),
                   reference_extrema(input, ir, kt, false),
                   0.f);

      // Public wrappers route to the same implementation.
      expect_close(hmap::gpu::local_max(input, ir, kt),
                   reference_extrema(input, ir, kt, true),
                   0.f);
      expect_close(hmap::gpu::local_min(input, ir, kt),
                   reference_extrema(input, ir, kt, false),
                   0.f);
    }
}

TEST(MetalRoutes, LocalExtremaDiskMatchesOpenCL)
{
  if (!metal_available()) GTEST_SKIP() << "Metal backend is not available";
  if (!opencl_available()) GTEST_SKIP() << "No OpenCL device for parity";

  Array input(glm::ivec2(53, 31));
  fill_field(input);
  for (const int ir : {1, 4, 11})
  {
    SCOPED_TRACE("ir=" + std::to_string(ir));
    expect_close(hmap::gpu::metal::local_max(input, ir, MinMaxKernel::DISK),
                 opencl_disk(input, ir, true),
                 0.f);
    expect_close(hmap::gpu::metal::local_min(input, ir, MinMaxKernel::DISK),
                 opencl_disk(input, ir, false),
                 0.f);
  }
}

TEST(MetalRoutes, SquareLocalMaxMatchesCpu)
{
  if (!metal_available()) GTEST_SKIP() << "Metal backend is not available";

  // The CPU local_max is the separable square (sliding window) filter.
  Array input(glm::ivec2(64, 33));
  fill_field(input);
  for (const int ir : {1, 3, 8})
  {
    SCOPED_TRACE("ir=" + std::to_string(ir));
    expect_close(hmap::gpu::local_max(input, ir, MinMaxKernel::SQUARE),
                 hmap::local_max(input, ir),
                 0.f);
    expect_close(hmap::gpu::local_min(input, ir, MinMaxKernel::SQUARE),
                 hmap::local_min(input, ir),
                 0.f);
  }
}

TEST(MetalRoutes, MorphologyOperatorsUseRoutedExtrema)
{
  if (!metal_available()) GTEST_SKIP() << "Metal backend is not available";

  Array input(glm::ivec2(45, 38));
  fill_field(input);
  const int ir = 4;

  for (const MinMaxKernel kt :
       {MinMaxKernel::DISK, MinMaxKernel::SQUARE, MinMaxKernel::OCTAGON})
  {
    SCOPED_TRACE(kernel_name(kt));
    const Array dil = reference_extrema(input, ir, kt, true);
    const Array ero = reference_extrema(input, ir, kt, false);

    expect_close(hmap::gpu::dilation(input, ir, kt), dil, 0.f);
    expect_close(hmap::gpu::erosion(input, ir, kt), ero, 0.f);
    expect_close(hmap::gpu::morphological_gradient(input, ir, kt),
                 dil - ero,
                 1e-5f);
    expect_close(hmap::gpu::opening(input, ir, kt),
                 reference_extrema(ero, ir, kt, true),
                 0.f);
    expect_close(hmap::gpu::closing(input, ir, kt),
                 reference_extrema(dil, ir, kt, false),
                 0.f);
  }
}

TEST(MetalRoutes, SmoothCpulseMatchesCpu)
{
  if (!metal_available()) GTEST_SKIP() << "Metal backend is not available";

  Array input(glm::ivec2(67, 29));
  fill_field(input);

  // 1100 exceeds the 4 KB setBytes limit and exercises the weight buffer.
  for (const int ir : {1, 4, 13, 64, 1100})
  {
    SCOPED_TRACE("ir=" + std::to_string(ir));
    Array expected = input;
    hmap::smooth_cpulse(expected, ir);

    Array actual = input;
    hmap::gpu::smooth_cpulse(actual, ir);
    expect_close(actual, expected, 2e-5f);
  }

  Array identity = input;
  hmap::gpu::smooth_cpulse(identity, 0);
  expect_close(identity, input, 0.f);
}

TEST(MetalRoutes, SmoothCpulseMaskedMatchesCpu)
{
  if (!metal_available()) GTEST_SKIP() << "Metal backend is not available";

  Array input(glm::ivec2(48, 40));
  fill_field(input);
  Array mask(input.shape);
  for (int j = 0; j < mask.shape.y; ++j)
    for (int i = 0; i < mask.shape.x; ++i)
      mask(i, j) = float(i) / float(mask.shape.x - 1);

  Array expected = input;
  hmap::smooth_cpulse(expected, 6, &mask);
  Array actual = input;
  hmap::gpu::smooth_cpulse(actual, 6, &mask);
  expect_close(actual, expected, 2e-5f);
}
