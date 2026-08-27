/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the LICENSE file. */

#include <cstdint>
#include <vector>

#include <benchmark/benchmark.h>

#include "cl_wrapper/run.hpp"

#include "highmap.hpp"

namespace
{

using hmap::Array;

const std::vector<int> k_pointwise_sizes = {
    128, 256, 512, 1024, 2048, 4096, 8192};
const std::vector<int> k_neighborhood_sizes = {128, 256, 512, 1024, 2048};

void apply_sizes(benchmark::internal::Benchmark *benchmark,
                 const std::vector<int> &sizes)
{
  for (const int size : sizes) benchmark->Arg(size);
}

Array make_field(int size)
{
  return hmap::white(glm::vec2(size, size), 0.f, 1.f, 42u);
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

Array opencl_maximum_smooth(const Array &array1, const Array &array2)
{
  Array out = array1;
  clwrapper::Run run("maximum_smooth");
  run.bind_buffer<float>("array1", out.vector);
  run.bind_buffer<float>("array2",
                         const_cast<std::vector<float> &>(array2.vector));
  run.bind_arguments(array1.shape.x, array1.shape.y, 0.2f);
  run.write_buffer("array1");
  run.write_buffer("array2");
  run.execute({array1.shape.x, array1.shape.y});
  run.read_buffer("array1");
  return out;
}

Array opencl_noise(const glm::ivec2 shape)
{
  Array out(shape);
  std::vector<float> dummy(1, 0.f);
  const glm::vec2 kw = {4.f, 4.f};
  const glm::vec4 bbox = {0.f, 1.f, 0.f, 1.f};

  clwrapper::Run run("noise");
  run.bind_buffer<float>("array", out.vector);
  run.bind_buffer<float>("noise_x", dummy);
  run.bind_buffer<float>("noise_y", dummy);
  run.bind_arguments(shape.x,
                     shape.y,
                     static_cast<int>(hmap::NoiseType::PERLIN),
                     kw.x,
                     kw.y,
                     42u,
                     0,
                     0,
                     0,
                     0,
                     bbox);
  run.write_buffer("array");
  run.execute({shape.x, shape.y});
  run.read_buffer("array");
  return out;
}

Array opencl_advection(const Array &z,
                       const Array &field,
                       const Array &dx,
                       const Array &dy)
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
  run.bind_arguments(z.shape.x, z.shape.y, 0.1f, 0.9f);
  run.execute({z.shape.x, z.shape.y});
  run.read_imagef("out");
  return out;
}

Array opencl_thermal(const Array &input, const Array &talus)
{
  Array out = input;
  clwrapper::Run run("thermal");
  run.bind_buffer<float>("z", out.vector);
  run.bind_buffer<float>("talus",
                         const_cast<std::vector<float> &>(talus.vector));
  run.bind_arguments(out.shape.x, out.shape.y, 0);
  run.write_buffer("z");
  run.write_buffer("talus");
  for (int iteration = 0; iteration < 4; ++iteration)
  {
    run.set_argument(4, iteration);
    run.execute({out.shape.x, out.shape.y});
  }
  run.read_buffer("z");
  return out;
}

void skip_if_metal_unavailable(benchmark::State &state)
{
  if (!hmap::gpu::metal::is_available())
    state.SkipWithError("Metal backend is unavailable on this host");
}

void skip_if_opencl_unavailable(benchmark::State &state)
{
  static const bool available = hmap::gpu::init_opencl();
  if (!available)
    state.SkipWithError("OpenCL backend is unavailable on this host");
}

void record_pixels(benchmark::State &state, int size)
{
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * size *
                          size);
}

template <typename Callable>
void warmup_once(benchmark::State &state, bool &warmed, Callable &&callable)
{
  if (warmed) return;
  state.PauseTiming();
  Array output = callable();
  benchmark::DoNotOptimize(output.vector.data());
  state.ResumeTiming();
  warmed = true;
}

} // namespace

static void BM_Apple_CPU_GradientNorm(benchmark::State &state)
{
  const int size = state.range(0);
  const Array input = make_field(size);
  bool warmed = false;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&] { return hmap::gradient_norm(input); });
    Array output = hmap::gradient_norm(input);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_pixels(state, size);
}

static void BM_Apple_OpenCL_GradientNorm(benchmark::State &state)
{
  skip_if_opencl_unavailable(state);
  if (state.skipped()) return;

  const int size = state.range(0);
  const Array input = make_field(size);
  bool warmed = false;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&] { return opencl_gradient_norm(input); });
    Array output = opencl_gradient_norm(input);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_pixels(state, size);
}

static void BM_Apple_CPU_SmoothMaximum(benchmark::State &state)
{
  const int size = state.range(0);
  const Array first = make_field(size);
  const Array second = make_field(size);
  bool warmed = false;
  for (auto _ : state)
  {
    warmup_once(state, warmed,
                [&] { return hmap::maximum_smooth(first, second, 0.2f); });
    Array output = hmap::maximum_smooth(first, second, 0.2f);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_pixels(state, size);
}

static void BM_Apple_OpenCL_SmoothMaximum(benchmark::State &state)
{
  skip_if_opencl_unavailable(state);
  if (state.skipped()) return;

  const int size = state.range(0);
  const Array first = make_field(size);
  const Array second = make_field(size);
  bool warmed = false;
  for (auto _ : state)
  {
    warmup_once(state, warmed,
                [&] { return opencl_maximum_smooth(first, second); });
    Array output = opencl_maximum_smooth(first, second);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_pixels(state, size);
}

static void BM_Apple_Metal_SmoothMaximum(benchmark::State &state)
{
  skip_if_metal_unavailable(state);
  if (state.skipped()) return;

  const int size = state.range(0);
  const Array first = make_field(size);
  const Array second = make_field(size);
  bool warmed = false;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&]
                {
                  return hmap::gpu::metal::maximum_smooth(
                      first, second, 0.2f);
                });
    Array output =
        hmap::gpu::metal::maximum_smooth(first, second, 0.2f);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_pixels(state, size);
}

static void BM_Apple_Metal_GradientNorm(benchmark::State &state)
{
  skip_if_metal_unavailable(state);
  if (state.skipped()) return;

  const int size = state.range(0);
  const Array input = make_field(size);
  bool warmed = false;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&]
                { return hmap::gpu::metal::gradient_norm(input); });
    Array output = hmap::gpu::metal::gradient_norm(input);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_pixels(state, size);
}

static void BM_Apple_CPU_Noise(benchmark::State &state)
{
  const int size = state.range(0);
  bool warmed = false;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&]
                { return hmap::noise(hmap::NoiseType::PERLIN,
                                     {size, size},
                                     {4.f, 4.f},
                                     42u); });
    Array output = hmap::noise(
        hmap::NoiseType::PERLIN, {size, size}, {4.f, 4.f}, 42u);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_pixels(state, size);
}

static void BM_Apple_OpenCL_Noise(benchmark::State &state)
{
  skip_if_opencl_unavailable(state);
  if (state.skipped()) return;

  const int size = state.range(0);
  bool warmed = false;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&] { return opencl_noise({size, size}); });
    Array output = opencl_noise({size, size});
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_pixels(state, size);
}

static void BM_Apple_Metal_Noise(benchmark::State &state)
{
  skip_if_metal_unavailable(state);
  if (state.skipped()) return;

  const int size = state.range(0);
  bool warmed = false;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&]
                { return hmap::gpu::metal::noise(
                      hmap::NoiseType::PERLIN,
                      {size, size},
                      {4.f, 4.f},
                      42u); });
    Array output = hmap::gpu::metal::noise(
        hmap::NoiseType::PERLIN, {size, size}, {4.f, 4.f}, 42u);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_pixels(state, size);
}

static void BM_Apple_OpenCL_Advection(benchmark::State &state)
{
  skip_if_opencl_unavailable(state);
  if (state.skipped()) return;

  const int size = state.range(0);
  const Array z = make_field(size);
  const Array field = make_field(size);
  const Array dx = hmap::gradient_x(z);
  const Array dy = hmap::gradient_y(z);
  bool warmed = false;
  for (auto _ : state)
  {
    warmup_once(state, warmed,
                [&] { return opencl_advection(z, field, dx, dy); });
    Array output = opencl_advection(z, field, dx, dy);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_pixels(state, size);
}

static void BM_Apple_Metal_Advection(benchmark::State &state)
{
  skip_if_metal_unavailable(state);
  if (state.skipped()) return;

  const int size = state.range(0);
  const Array z = make_field(size);
  const Array field = make_field(size);
  const Array dx = hmap::gradient_x(z);
  const Array dy = hmap::gradient_y(z);
  bool warmed = false;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&]
                { return hmap::gpu::metal::advection_warp(
                      z, field, dx, dy, 0.1f, 0.9f); });
    Array output = hmap::gpu::metal::advection_warp(
        z, field, dx, dy, 0.1f, 0.9f);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_pixels(state, size);
}

static void BM_Apple_OpenCL_Thermal(benchmark::State &state)
{
  skip_if_opencl_unavailable(state);
  if (state.skipped()) return;

  const int size = state.range(0);
  const Array input = make_field(size);
  const Array talus(input.shape, 0.01f);
  bool warmed = false;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&] { return opencl_thermal(input, talus); });
    Array output = opencl_thermal(input, talus);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_pixels(state, size);
}

static void BM_Apple_Metal_Thermal(benchmark::State &state)
{
  skip_if_metal_unavailable(state);
  if (state.skipped()) return;

  const int size = state.range(0);
  const Array input = make_field(size);
  const Array talus(input.shape, 0.01f);
  bool warmed = false;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&]
                {
                  Array output = input;
                  hmap::gpu::metal::thermal(output, talus, 4);
                  return output;
                });
    Array output = input;
    hmap::gpu::metal::thermal(output, talus, 4);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_pixels(state, size);
}

static void BM_Apple_Metal_HydraulicVPipes(benchmark::State &state)
{
  skip_if_metal_unavailable(state);
  if (state.skipped()) return;

  const int size = state.range(0);
  const Array input = make_field(size);
  bool warmed = false;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&]
                {
                  Array output = input;
                  hmap::gpu::metal::hydraulic_vpipes(output,
                                                     0.01f,
                                                     true,
                                                     0.1f,
                                                     4,
                                                     0.5f,
                                                     0.5f,
                                                     0.001f,
                                                     0.01f,
                                                     1.f,
                                                     10.f,
                                                     true,
                                                     0.01f,
                                                     nullptr,
                                                     nullptr,
                                                     nullptr,
                                                     nullptr,
                                                     nullptr);
                  return output;
                });
    Array output = input;
    hmap::gpu::metal::hydraulic_vpipes(output,
                                       0.01f,
                                       true,
                                       0.1f,
                                       4,
                                       0.5f,
                                       0.5f,
                                       0.001f,
                                       0.01f,
                                       1.f,
                                       10.f,
                                       true,
                                       0.01f,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       nullptr);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_pixels(state, size);
}

BENCHMARK(BM_Apple_CPU_GradientNorm)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_sizes(b, k_pointwise_sizes); })
    ->UseRealTime();
BENCHMARK(BM_Apple_OpenCL_GradientNorm)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_sizes(b, k_pointwise_sizes); })
    ->UseRealTime();
BENCHMARK(BM_Apple_Metal_GradientNorm)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_sizes(b, k_pointwise_sizes); })
    ->UseRealTime();

BENCHMARK(BM_Apple_CPU_SmoothMaximum)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_sizes(b, k_pointwise_sizes); })
    ->UseRealTime();
BENCHMARK(BM_Apple_OpenCL_SmoothMaximum)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_sizes(b, k_pointwise_sizes); })
    ->UseRealTime();
BENCHMARK(BM_Apple_Metal_SmoothMaximum)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_sizes(b, k_pointwise_sizes); })
    ->UseRealTime();

BENCHMARK(BM_Apple_CPU_Noise)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_sizes(b, k_pointwise_sizes); })
    ->UseRealTime();
BENCHMARK(BM_Apple_OpenCL_Noise)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_sizes(b, k_pointwise_sizes); })
    ->UseRealTime();
BENCHMARK(BM_Apple_Metal_Noise)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_sizes(b, k_pointwise_sizes); })
    ->UseRealTime();

BENCHMARK(BM_Apple_OpenCL_Advection)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_sizes(b, k_neighborhood_sizes); })
    ->UseRealTime();
BENCHMARK(BM_Apple_Metal_Advection)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_sizes(b, k_neighborhood_sizes); })
    ->UseRealTime();

BENCHMARK(BM_Apple_OpenCL_Thermal)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_sizes(b, k_neighborhood_sizes); })
    ->UseRealTime();
BENCHMARK(BM_Apple_Metal_Thermal)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_sizes(b, k_neighborhood_sizes); })
    ->UseRealTime();

BENCHMARK(BM_Apple_Metal_HydraulicVPipes)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_sizes(b, k_neighborhood_sizes); })
    ->UseRealTime();
