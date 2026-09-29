/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the LICENSE file. */

// Metal vs OpenCL for the public hmap::gpu wrappers routed to Metal after the
// upstream sync. The OpenCL side calls the kernels directly (the wrappers
// themselves now dispatch to Metal when it is available), replicating the
// upstream host code including its host round trips.

#include <cmath>
#include <vector>

#include <benchmark/benchmark.h>

#include "highmap/internal/opencl_run.hpp"

#include "highmap.hpp"
#include "highmap/gpu/metal.hpp"

namespace
{

using hmap::Array;
using hmap::MinMaxKernel;

Array route_field(int size)
{
  return hmap::white(glm::vec2(size, size), 0.f, 1.f, 42u);
}

bool route_skip_opencl(benchmark::State &state)
{
  static const bool available = hmap::gpu::init_opencl();
  if (!available) state.SkipWithError("OpenCL backend is unavailable");
  return !available;
}

bool route_skip_metal(benchmark::State &state)
{
  if (!hmap::gpu::metal::is_available())
  {
    state.SkipWithError("Metal backend is unavailable");
    return true;
  }
  return false;
}

void route_pixels(benchmark::State &state, int size)
{
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * size *
                          size);
}

Array opencl_smooth_cpulse(Array array, int ir)
{
  const int          nk = 2 * ir + 1;
  std::vector<float> k1d(nk);
  float              sum = 0.f;
  for (int i = 0; i < nk; i++)
  {
    const float x = std::abs(float(i) - float(ir)) / float(ir);
    k1d[i] = std::exp(-0.5f * x * x * 9.f);
    sum += k1d[i];
  }
  for (float &w : k1d) w /= sum;

  auto run = clwrapper::Run("smooth_cpulse");
  run.bind_imagef("in", array.vector, array.shape.x, array.shape.y);
  run.bind_imagef("weights", k1d, nk, 1);
  run.bind_imagef("out", array.vector, array.shape.x, array.shape.y, true);
  run.bind_arguments(array.shape.x, array.shape.y, ir);
  run.set_argument(6, 0);
  run.execute({array.shape.x, array.shape.y});
  run.read_imagef("out");
  run.write_imagef("in");
  run.set_argument(6, 1);
  run.execute({array.shape.x, array.shape.y});
  run.read_imagef("out");
  return array;
}

Array opencl_local_max_disk(const Array &array, int ir)
{
  Array              out(array.shape);
  std::vector<float> in = array.vector;
  auto               run = clwrapper::Run("local_max");
  run.bind_imagef("array", in, array.shape.x, array.shape.y);
  run.bind_imagef("out", out.vector, array.shape.x, array.shape.y, true);
  run.bind_arguments(array.shape.x, array.shape.y, ir);
  run.execute({array.shape.x, array.shape.y});
  run.read_imagef("out");
  return out;
}

Array opencl_local_max_octagon(Array array, int ir)
{
  const int b = static_cast<int>(
      std::round((std::sqrt(2.f) - 1.f) * static_cast<float>(ir)));
  const int a = ir - b;
  auto      run = clwrapper::Run("local_max_octagon");
  run.bind_imagef("in", array.vector, array.shape.x, array.shape.y);
  run.bind_imagef("out", array.vector, array.shape.x, array.shape.y, true);
  run.bind_arguments(array.shape.x, array.shape.y, a, 0);
  const int radii[4] = {a, a, b, b};
  for (int pass = 0; pass < (b > 0 ? 4 : 2); ++pass)
  {
    if (pass > 0) run.write_imagef("in");
    run.set_argument(4, radii[pass]);
    run.set_argument(5, pass);
    run.execute({array.shape.x, array.shape.y});
    run.read_imagef("out");
  }
  return array;
}

constexpr int k_route_ir = 16;

} // namespace

static void BM_Route_OpenCL_SmoothCpulse(benchmark::State &state)
{
  if (route_skip_opencl(state)) return;
  const int   size = int(state.range(0));
  const Array input = route_field(size);
  for (auto _ : state)
  {
    Array out = opencl_smooth_cpulse(input, k_route_ir);
    benchmark::DoNotOptimize(out.vector.data());
  }
  route_pixels(state, size);
}

static void BM_Route_Metal_SmoothCpulse(benchmark::State &state)
{
  if (route_skip_metal(state)) return;
  const int   size = int(state.range(0));
  const Array input = route_field(size);
  for (auto _ : state)
  {
    Array out = input;
    hmap::gpu::smooth_cpulse(out, k_route_ir);
    benchmark::DoNotOptimize(out.vector.data());
  }
  route_pixels(state, size);
}

static void BM_Route_OpenCL_LocalMaxDisk(benchmark::State &state)
{
  if (route_skip_opencl(state)) return;
  const int   size = int(state.range(0));
  const Array input = route_field(size);
  for (auto _ : state)
  {
    Array out = opencl_local_max_disk(input, k_route_ir);
    benchmark::DoNotOptimize(out.vector.data());
  }
  route_pixels(state, size);
}

static void BM_Route_Metal_LocalMaxDisk(benchmark::State &state)
{
  if (route_skip_metal(state)) return;
  const int   size = int(state.range(0));
  const Array input = route_field(size);
  for (auto _ : state)
  {
    Array out = hmap::gpu::local_max(input, k_route_ir, MinMaxKernel::DISK);
    benchmark::DoNotOptimize(out.vector.data());
  }
  route_pixels(state, size);
}

static void BM_Route_OpenCL_LocalMaxOctagon(benchmark::State &state)
{
  if (route_skip_opencl(state)) return;
  const int   size = int(state.range(0));
  const Array input = route_field(size);
  for (auto _ : state)
  {
    Array out = opencl_local_max_octagon(input, k_route_ir);
    benchmark::DoNotOptimize(out.vector.data());
  }
  route_pixels(state, size);
}

static void BM_Route_Metal_LocalMaxOctagon(benchmark::State &state)
{
  if (route_skip_metal(state)) return;
  const int   size = int(state.range(0));
  const Array input = route_field(size);
  for (auto _ : state)
  {
    Array out = hmap::gpu::local_max(input, k_route_ir, MinMaxKernel::OCTAGON);
    benchmark::DoNotOptimize(out.vector.data());
  }
  route_pixels(state, size);
}

#define HMAP_ROUTE_SIZES ->Arg(1024)->Arg(2048)->Arg(4096)->UseRealTime()

BENCHMARK(BM_Route_OpenCL_SmoothCpulse) HMAP_ROUTE_SIZES;
BENCHMARK(BM_Route_Metal_SmoothCpulse) HMAP_ROUTE_SIZES;
BENCHMARK(BM_Route_OpenCL_LocalMaxDisk) HMAP_ROUTE_SIZES;
BENCHMARK(BM_Route_Metal_LocalMaxDisk) HMAP_ROUTE_SIZES;
BENCHMARK(BM_Route_OpenCL_LocalMaxOctagon) HMAP_ROUTE_SIZES;
BENCHMARK(BM_Route_Metal_LocalMaxOctagon) HMAP_ROUTE_SIZES;
