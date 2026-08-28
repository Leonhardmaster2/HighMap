/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU
 * General Public License. The full license is in the LICENSE file. */

#include <chrono>
#include <cstdint>
#include <vector>

#include <benchmark/benchmark.h>

#include "cl_wrapper/run.hpp"

#include "highmap.hpp"

namespace
{

using hmap::Array;
using Clock = std::chrono::steady_clock;

const std::vector<int> k_pointwise_sizes = {
    128, 256, 512, 1024, 2048, 4096, 8192};
const std::vector<int> k_neighborhood_sizes = {
    128, 256, 512, 1024, 2048, 4096};
const std::vector<int> k_iterations = {10, 50, 100, 250};

struct TimingBreakdown
{
  double total_ms = 0.0;
  double allocation_ms = 0.0;
  double upload_ms = 0.0;
  double pipeline_lookup_ms = 0.0;
  double encoding_ms = 0.0;
  double device_or_finish_ms = 0.0;
  double synchronization_ms = 0.0;
  double readback_ms = 0.0;
  double host_compute_ms = 0.0;
};

double elapsed_ms(Clock::time_point start)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

void apply_sizes(benchmark::internal::Benchmark *benchmark,
                 const std::vector<int> &sizes)
{
  for (const int size : sizes) benchmark->Arg(size);
}

void apply_size_iterations(benchmark::internal::Benchmark *benchmark,
                           const std::vector<int> &sizes)
{
  for (const int size : sizes)
    for (const int iterations : k_iterations)
      benchmark->Args({size, iterations});
}

Array make_field(int size)
{
  return hmap::white(glm::vec2(size, size), 0.f, 1.f, 42u);
}

void record_opencl_finish(TimingBreakdown *timing, float kernel_ms)
{
  if (!timing) return;
  // CLWrapper exposes a blocking queue-finish duration rather than a device
  // timestamp. Keep both names explicit so reports do not mistake this for a
  // pure GPU execution timestamp.
  timing->device_or_finish_ms += kernel_ms;
  timing->synchronization_ms += kernel_ms;
}

Array opencl_gradient_norm(const Array &input, TimingBreakdown *timing = nullptr)
{
  const auto total_start = Clock::now();
  Array out(input.shape);

  auto pipeline_start = Clock::now();
  clwrapper::Run run("gradient_norm");
  if (timing) timing->pipeline_lookup_ms += elapsed_ms(pipeline_start);

  auto allocation_start = Clock::now();
  run.bind_buffer<float>("array",
                         const_cast<std::vector<float> &>(input.vector));
  run.bind_buffer<float>("dm", out.vector);
  run.bind_arguments(input.shape.x, input.shape.y);
  if (timing) timing->allocation_ms += elapsed_ms(allocation_start);

  auto upload_start = Clock::now();
  run.write_buffer("array");
  if (timing) timing->upload_ms += elapsed_ms(upload_start);

  float kernel_ms = 0.f;
  run.execute({input.shape.x, input.shape.y}, &kernel_ms);
  record_opencl_finish(timing, kernel_ms);

  auto readback_start = Clock::now();
  run.read_buffer("dm");
  if (timing) timing->readback_ms += elapsed_ms(readback_start);

  if (timing) timing->total_ms = elapsed_ms(total_start);
  return out;
}

Array opencl_maximum_smooth(const Array       &array1,
                            const Array       &array2,
                            TimingBreakdown *timing = nullptr)
{
  const auto total_start = Clock::now();
  Array out = array1;

  auto pipeline_start = Clock::now();
  clwrapper::Run run("maximum_smooth");
  if (timing) timing->pipeline_lookup_ms += elapsed_ms(pipeline_start);

  auto allocation_start = Clock::now();
  run.bind_buffer<float>("array1", out.vector);
  run.bind_buffer<float>("array2",
                         const_cast<std::vector<float> &>(array2.vector));
  run.bind_arguments(array1.shape.x, array1.shape.y, 0.2f);
  if (timing) timing->allocation_ms += elapsed_ms(allocation_start);

  auto upload_start = Clock::now();
  run.write_buffer("array1");
  run.write_buffer("array2");
  if (timing) timing->upload_ms += elapsed_ms(upload_start);

  float kernel_ms = 0.f;
  run.execute({array1.shape.x, array1.shape.y}, &kernel_ms);
  record_opencl_finish(timing, kernel_ms);

  auto readback_start = Clock::now();
  run.read_buffer("array1");
  if (timing) timing->readback_ms += elapsed_ms(readback_start);

  if (timing) timing->total_ms = elapsed_ms(total_start);
  return out;
}

Array opencl_noise(const glm::ivec2 shape, TimingBreakdown *timing = nullptr)
{
  const auto total_start = Clock::now();
  Array out(shape);
  std::vector<float> dummy(1, 0.f);
  const glm::vec2 kw = {4.f, 4.f};
  const glm::vec4 bbox = {0.f, 1.f, 0.f, 1.f};

  auto pipeline_start = Clock::now();
  clwrapper::Run run("noise");
  if (timing) timing->pipeline_lookup_ms += elapsed_ms(pipeline_start);

  auto allocation_start = Clock::now();
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
  if (timing) timing->allocation_ms += elapsed_ms(allocation_start);

  auto upload_start = Clock::now();
  run.write_buffer("array");
  if (timing) timing->upload_ms += elapsed_ms(upload_start);

  float kernel_ms = 0.f;
  run.execute({shape.x, shape.y}, &kernel_ms);
  record_opencl_finish(timing, kernel_ms);

  auto readback_start = Clock::now();
  run.read_buffer("array");
  if (timing) timing->readback_ms += elapsed_ms(readback_start);

  if (timing) timing->total_ms = elapsed_ms(total_start);
  return out;
}

Array opencl_advection(const Array       &z,
                       const Array       &field,
                       const Array       &dx,
                       const Array       &dy,
                       TimingBreakdown *timing = nullptr)
{
  const auto total_start = Clock::now();
  Array out(z.shape);
  Array mask(z.shape, 1.f);

  auto pipeline_start = Clock::now();
  clwrapper::Run run("advection_warp");
  if (timing) timing->pipeline_lookup_ms += elapsed_ms(pipeline_start);

  // OpenCL image bindings use CL_MEM_COPY_HOST_PTR for inputs. That copy is
  // intentionally classified as preparation/upload rather than hidden in the
  // kernel duration.
  auto upload_start = Clock::now();
  run.bind_imagef("z", z.vector, z.shape.x, z.shape.y);
  run.bind_imagef("field", field.vector, z.shape.x, z.shape.y);
  run.bind_imagef("dx", dx.vector, z.shape.x, z.shape.y);
  run.bind_imagef("dy", dy.vector, z.shape.x, z.shape.y);
  run.bind_imagef("mask", mask.vector, z.shape.x, z.shape.y);
  if (timing) timing->upload_ms += elapsed_ms(upload_start);

  auto allocation_start = Clock::now();
  run.bind_imagef("out", out.vector, z.shape.x, z.shape.y, true);
  run.bind_arguments(z.shape.x, z.shape.y, 0.1f, 0.9f);
  if (timing) timing->allocation_ms += elapsed_ms(allocation_start);

  float kernel_ms = 0.f;
  run.execute({z.shape.x, z.shape.y}, &kernel_ms);
  record_opencl_finish(timing, kernel_ms);

  auto readback_start = Clock::now();
  run.read_imagef("out");
  if (timing) timing->readback_ms += elapsed_ms(readback_start);

  if (timing) timing->total_ms = elapsed_ms(total_start);
  return out;
}

Array opencl_thermal(const Array       &input,
                     const Array       &talus,
                     int                iterations,
                     TimingBreakdown *timing = nullptr)
{
  const auto total_start = Clock::now();
  Array out = input;

  auto pipeline_start = Clock::now();
  clwrapper::Run run("thermal");
  if (timing) timing->pipeline_lookup_ms += elapsed_ms(pipeline_start);

  auto allocation_start = Clock::now();
  run.bind_buffer<float>("z", out.vector);
  run.bind_buffer<float>("talus",
                         const_cast<std::vector<float> &>(talus.vector));
  run.bind_arguments(out.shape.x, out.shape.y, 0);
  if (timing) timing->allocation_ms += elapsed_ms(allocation_start);

  auto upload_start = Clock::now();
  run.write_buffer("z");
  run.write_buffer("talus");
  if (timing) timing->upload_ms += elapsed_ms(upload_start);

  for (int iteration = 0; iteration < iterations; ++iteration)
  {
    run.set_argument(4, iteration);
    float kernel_ms = 0.f;
    run.execute({out.shape.x, out.shape.y}, &kernel_ms);
    record_opencl_finish(timing, kernel_ms);
  }

  auto readback_start = Clock::now();
  run.read_buffer("z");
  if (timing) timing->readback_ms += elapsed_ms(readback_start);

  if (timing) timing->total_ms = elapsed_ms(total_start);
  return out;
}

// This is deliberately a benchmark-only copy of the existing OpenCL
// implementation. It makes the CPU participation in that path measurable:
// every pass creates an image wrapper, blocks for completion, and copies its
// result back before the next pass consumes it.
Array opencl_hydraulic_vpipes(const Array       &input,
                              int                iterations,
                              TimingBreakdown *timing = nullptr)
{
  const auto total_start = Clock::now();
  const glm::ivec2 shape = input.shape;
  const float water_height = 0.01f;
  const bool maintain_water_volume = true;
  const float evap_rate = 0.1f;
  const float dt = 0.5f;
  const float k_capacity = 0.5f;
  const float k_erode = 0.001f;
  const float k_depose = 0.01f;
  const float k_discharge_exp = 1.f;
  const float downcutting_max_depth_ratio = 10.f;
  const bool flux_diffusion = true;
  const float flux_diffusion_strength = 0.01f;

  Array z = input;
  Array rain_map(shape, 1.f);
  Array d(shape, water_height);
  Array d1(shape);
  Array d2(shape);
  Array s(shape);
  Array fl(shape);
  Array fr(shape);
  Array ft(shape);
  Array fb(shape);
  Array u(shape);
  Array v(shape);
  d *= rain_map;
  const float water_volume_init = d.sum();
  const float rain_map_volume = rain_map.sum();

  for (int it = 0; it < iterations; ++it)
  {
    auto host_start = Clock::now();
    d1 = d;
    if (timing) timing->host_compute_ms += elapsed_ms(host_start);

    {
      auto pipeline_start = Clock::now();
      clwrapper::Run run("hydraulic_vpipes_flow_pass");
      if (timing) timing->pipeline_lookup_ms += elapsed_ms(pipeline_start);
      auto allocation_start = Clock::now();
      run.bind_imagef("z", z.vector, shape.x, shape.y);
      run.bind_imagef("fl", fl.vector, shape.x, shape.y);
      run.bind_imagef("fr", fr.vector, shape.x, shape.y);
      run.bind_imagef("ft", ft.vector, shape.x, shape.y);
      run.bind_imagef("fb", fb.vector, shape.x, shape.y);
      run.bind_imagef("d1", d1.vector, shape.x, shape.y);
      run.bind_imagef("fl_out", fl.vector, shape.x, shape.y, true);
      run.bind_imagef("fr_out", fr.vector, shape.x, shape.y, true);
      run.bind_imagef("ft_out", ft.vector, shape.x, shape.y, true);
      run.bind_imagef("fb_out", fb.vector, shape.x, shape.y, true);
      run.bind_arguments(shape.x,
                         shape.y,
                         dt,
                         flux_diffusion ? 1 : 0,
                         flux_diffusion_strength);
      if (timing) timing->allocation_ms += elapsed_ms(allocation_start);
      float kernel_ms = 0.f;
      run.execute({shape.x, shape.y}, &kernel_ms);
      record_opencl_finish(timing, kernel_ms);
      auto readback_start = Clock::now();
      run.read_imagef("fl_out");
      run.read_imagef("fr_out");
      run.read_imagef("ft_out");
      run.read_imagef("fb_out");
      if (timing) timing->readback_ms += elapsed_ms(readback_start);
    }

    {
      auto pipeline_start = Clock::now();
      clwrapper::Run run("hydraulic_vpipes_water_pass");
      if (timing) timing->pipeline_lookup_ms += elapsed_ms(pipeline_start);
      auto allocation_start = Clock::now();
      run.bind_imagef("z", z.vector, shape.x, shape.y);
      run.bind_imagef("fl", fl.vector, shape.x, shape.y);
      run.bind_imagef("fr", fr.vector, shape.x, shape.y);
      run.bind_imagef("ft", ft.vector, shape.x, shape.y);
      run.bind_imagef("fb", fb.vector, shape.x, shape.y);
      run.bind_imagef("d1", d1.vector, shape.x, shape.y);
      run.bind_imagef("d2_out", d2.vector, shape.x, shape.y, true);
      run.bind_imagef("u_out", u.vector, shape.x, shape.y, true);
      run.bind_imagef("v_out", v.vector, shape.x, shape.y, true);
      run.bind_arguments(shape.x, shape.y, dt, water_height);
      if (timing) timing->allocation_ms += elapsed_ms(allocation_start);
      float kernel_ms = 0.f;
      run.execute({shape.x, shape.y}, &kernel_ms);
      record_opencl_finish(timing, kernel_ms);
      auto readback_start = Clock::now();
      run.read_imagef("d2_out");
      run.read_imagef("u_out");
      run.read_imagef("v_out");
      if (timing) timing->readback_ms += elapsed_ms(readback_start);
    }

    {
      auto pipeline_start = Clock::now();
      clwrapper::Run run("hydraulic_vpipes_erosion_pass");
      if (timing) timing->pipeline_lookup_ms += elapsed_ms(pipeline_start);
      auto allocation_start = Clock::now();
      run.bind_imagef("z", z.vector, shape.x, shape.y);
      run.bind_imagef("d2", d2.vector, shape.x, shape.y);
      run.bind_imagef("u", u.vector, shape.x, shape.y);
      run.bind_imagef("v", v.vector, shape.x, shape.y);
      run.bind_imagef("s", s.vector, shape.x, shape.y);
      run.bind_imagef("z_out", z.vector, shape.x, shape.y, true);
      run.bind_imagef("s_out", s.vector, shape.x, shape.y, true);
      run.bind_arguments(shape.x,
                         shape.y,
                         water_height,
                         k_capacity,
                         k_erode,
                         k_depose,
                         k_discharge_exp,
                         downcutting_max_depth_ratio);
      if (timing) timing->allocation_ms += elapsed_ms(allocation_start);
      float kernel_ms = 0.f;
      run.execute({shape.x, shape.y}, &kernel_ms);
      record_opencl_finish(timing, kernel_ms);
      auto readback_start = Clock::now();
      run.read_imagef("z_out");
      run.read_imagef("s_out");
      if (timing) timing->readback_ms += elapsed_ms(readback_start);
    }

    {
      auto pipeline_start = Clock::now();
      clwrapper::Run run("hydraulic_vpipes_sediment_transport_pass");
      if (timing) timing->pipeline_lookup_ms += elapsed_ms(pipeline_start);
      auto allocation_start = Clock::now();
      run.bind_imagef("u", u.vector, shape.x, shape.y);
      run.bind_imagef("v", v.vector, shape.x, shape.y);
      run.bind_imagef("s", s.vector, shape.x, shape.y);
      run.bind_imagef("s_out", s.vector, shape.x, shape.y, true);
      run.bind_arguments(shape.x, shape.y, dt);
      if (timing) timing->allocation_ms += elapsed_ms(allocation_start);
      float kernel_ms = 0.f;
      run.execute({shape.x, shape.y}, &kernel_ms);
      record_opencl_finish(timing, kernel_ms);
      auto readback_start = Clock::now();
      run.read_imagef("s_out");
      if (timing) timing->readback_ms += elapsed_ms(readback_start);
    }

    auto host_finish_start = Clock::now();
    d = d2 * (1.f - dt * evap_rate);
    if (maintain_water_volume)
    {
      const float water_to_add = water_volume_init - d.sum();
      d += water_to_add * rain_map / rain_map_volume;
    }
    if (timing) timing->host_compute_ms += elapsed_ms(host_finish_start);
  }

  if (timing) timing->total_ms = elapsed_ms(total_start);
  return z;
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

void record_pixels(benchmark::State &state, int size, int iterations = 1)
{
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * size *
                          size * iterations);
}

void record_timing(benchmark::State &state, const TimingBreakdown &timing)
{
  state.counters["alloc_ms"] = timing.allocation_ms;
  state.counters["upload_ms"] = timing.upload_ms;
  state.counters["pipeline_ms"] = timing.pipeline_lookup_ms;
  state.counters["encode_ms"] = timing.encoding_ms;
  state.counters["device_or_finish_ms"] = timing.device_or_finish_ms;
  state.counters["sync_ms"] = timing.synchronization_ms;
  state.counters["readback_ms"] = timing.readback_ms;
  state.counters["host_between_pass_ms"] = timing.host_compute_ms;
  state.counters["measured_total_ms"] = timing.total_ms;
}

void record_metal_timing(benchmark::State                  &state,
                         const TimingBreakdown             &timing,
                         const hmap::gpu::metal::ExecutionStats &stats)
{
  TimingBreakdown measured = timing;
  measured.allocation_ms = stats.allocation_ms;
  measured.upload_ms = stats.upload_ms;
  measured.pipeline_lookup_ms = stats.pipeline_lookup_ms;
  measured.encoding_ms = stats.encoding_ms;
  measured.device_or_finish_ms = stats.gpu_execution_ms;
  measured.synchronization_ms = stats.wait_ms;
  measured.readback_ms = stats.readback_ms;
  record_timing(state, measured);
  state.counters["buffer_allocations"] =
      static_cast<double>(stats.buffer_allocations);
  state.counters["pipeline_creations"] =
      static_cast<double>(stats.pipeline_creations);
  state.counters["command_buffers"] =
      static_cast<double>(stats.command_buffers);
  state.counters["encoders"] = static_cast<double>(stats.encoders);
  state.counters["upload_bytes"] = static_cast<double>(stats.upload_bytes);
  state.counters["readback_bytes"] = static_cast<double>(stats.readback_bytes);
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
  TimingBreakdown timing;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&] { return opencl_gradient_norm(input); });
    timing = {};
    Array output = opencl_gradient_norm(input, &timing);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_timing(state, timing);
  record_pixels(state, size);
}

static void BM_Apple_Metal_GradientNorm(benchmark::State &state)
{
  skip_if_metal_unavailable(state);
  if (state.skipped()) return;
  const int size = state.range(0);
  const Array input = make_field(size);
  bool warmed = false;
  TimingBreakdown timing;
  hmap::gpu::metal::ExecutionStats stats;
  for (auto _ : state)
  {
    warmup_once(state, warmed,
                [&] { return hmap::gpu::metal::gradient_norm(input); });
    const auto total_start = Clock::now();
    Array output = hmap::gpu::metal::gradient_norm(input);
    timing.total_ms = elapsed_ms(total_start);
    stats = hmap::gpu::metal::last_execution_stats();
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_metal_timing(state, timing, stats);
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
  TimingBreakdown timing;
  for (auto _ : state)
  {
    warmup_once(state, warmed,
                [&] { return opencl_maximum_smooth(first, second); });
    timing = {};
    Array output = opencl_maximum_smooth(first, second, &timing);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_timing(state, timing);
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
  TimingBreakdown timing;
  hmap::gpu::metal::ExecutionStats stats;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&]
                {
                  return hmap::gpu::metal::maximum_smooth(
                      first, second, 0.2f);
                });
    const auto total_start = Clock::now();
    Array output =
        hmap::gpu::metal::maximum_smooth(first, second, 0.2f);
    timing.total_ms = elapsed_ms(total_start);
    stats = hmap::gpu::metal::last_execution_stats();
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_metal_timing(state, timing, stats);
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
  TimingBreakdown timing;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&] { return opencl_noise({size, size}); });
    timing = {};
    Array output = opencl_noise({size, size}, &timing);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_timing(state, timing);
  record_pixels(state, size);
}

static void BM_Apple_Metal_Noise(benchmark::State &state)
{
  skip_if_metal_unavailable(state);
  if (state.skipped()) return;
  const int size = state.range(0);
  bool warmed = false;
  TimingBreakdown timing;
  hmap::gpu::metal::ExecutionStats stats;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&]
                { return hmap::gpu::metal::noise(
                      hmap::NoiseType::PERLIN,
                      {size, size},
                      {4.f, 4.f},
                      42u); });
    const auto total_start = Clock::now();
    Array output = hmap::gpu::metal::noise(
        hmap::NoiseType::PERLIN, {size, size}, {4.f, 4.f}, 42u);
    timing.total_ms = elapsed_ms(total_start);
    stats = hmap::gpu::metal::last_execution_stats();
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_metal_timing(state, timing, stats);
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
  TimingBreakdown timing;
  for (auto _ : state)
  {
    warmup_once(state, warmed,
                [&] { return opencl_advection(z, field, dx, dy); });
    timing = {};
    Array output = opencl_advection(z, field, dx, dy, &timing);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_timing(state, timing);
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
  TimingBreakdown timing;
  hmap::gpu::metal::ExecutionStats stats;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&]
                { return hmap::gpu::metal::advection_warp(
                      z, field, dx, dy, 0.1f, 0.9f); });
    const auto total_start = Clock::now();
    Array output = hmap::gpu::metal::advection_warp(
        z, field, dx, dy, 0.1f, 0.9f);
    timing.total_ms = elapsed_ms(total_start);
    stats = hmap::gpu::metal::last_execution_stats();
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_metal_timing(state, timing, stats);
  record_pixels(state, size);
}

static void BM_Apple_OpenCL_Thermal(benchmark::State &state)
{
  skip_if_opencl_unavailable(state);
  if (state.skipped()) return;
  const int size = state.range(0);
  const int iterations = state.range(1);
  const Array input = make_field(size);
  const Array talus(input.shape, 0.01f);
  bool warmed = false;
  TimingBreakdown timing;
  for (auto _ : state)
  {
    warmup_once(state, warmed,
                [&] { return opencl_thermal(input, talus, iterations); });
    timing = {};
    Array output = opencl_thermal(input, talus, iterations, &timing);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_timing(state, timing);
  record_pixels(state, size, iterations);
}

static void BM_Apple_Metal_Thermal(benchmark::State &state)
{
  skip_if_metal_unavailable(state);
  if (state.skipped()) return;
  const int size = state.range(0);
  const int iterations = state.range(1);
  const Array input = make_field(size);
  const Array talus(input.shape, 0.01f);
  bool warmed = false;
  TimingBreakdown timing;
  hmap::gpu::metal::ExecutionStats stats;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&]
                {
                  Array output = input;
                  hmap::gpu::metal::thermal(output, talus, iterations);
                  return output;
                });
    const auto total_start = Clock::now();
    Array output = input;
    hmap::gpu::metal::thermal(output, talus, iterations);
    timing.total_ms = elapsed_ms(total_start);
    stats = hmap::gpu::metal::last_execution_stats();
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_metal_timing(state, timing, stats);
  record_pixels(state, size, iterations);
}

static void BM_Apple_OpenCL_HydraulicVPipes(benchmark::State &state)
{
  skip_if_opencl_unavailable(state);
  if (state.skipped()) return;
  const int size = state.range(0);
  const int iterations = state.range(1);
  const Array input = make_field(size);
  bool warmed = false;
  TimingBreakdown timing;
  for (auto _ : state)
  {
    warmup_once(state, warmed,
                [&] { return opencl_hydraulic_vpipes(input, iterations); });
    timing = {};
    Array output = opencl_hydraulic_vpipes(input, iterations, &timing);
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_timing(state, timing);
  record_pixels(state, size, iterations);
}

static void BM_Apple_Metal_HydraulicVPipes(benchmark::State &state)
{
  skip_if_metal_unavailable(state);
  if (state.skipped()) return;
  const int size = state.range(0);
  const int iterations = state.range(1);
  const Array input = make_field(size);
  bool warmed = false;
  TimingBreakdown timing;
  hmap::gpu::metal::ExecutionStats stats;
  for (auto _ : state)
  {
    warmup_once(state, warmed, [&]
                {
                  Array output = input;
                  hmap::gpu::metal::hydraulic_vpipes(output,
                                                     0.01f,
                                                     true,
                                                     0.1f,
                                                     iterations,
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
    const auto total_start = Clock::now();
    Array output = input;
    hmap::gpu::metal::hydraulic_vpipes(output,
                                       0.01f,
                                       true,
                                       0.1f,
                                       iterations,
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
    timing.total_ms = elapsed_ms(total_start);
    stats = hmap::gpu::metal::last_execution_stats();
    benchmark::DoNotOptimize(output.vector.data());
  }
  record_metal_timing(state, timing, stats);
  record_pixels(state, size, iterations);
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
            { apply_size_iterations(b, k_neighborhood_sizes); })
    ->UseRealTime();
BENCHMARK(BM_Apple_Metal_Thermal)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_size_iterations(b, k_neighborhood_sizes); })
    ->UseRealTime();

BENCHMARK(BM_Apple_OpenCL_HydraulicVPipes)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_size_iterations(b, k_neighborhood_sizes); })
    ->UseRealTime();
BENCHMARK(BM_Apple_Metal_HydraulicVPipes)
    ->Apply([](benchmark::internal::Benchmark *b)
            { apply_size_iterations(b, k_neighborhood_sizes); })
    ->UseRealTime();
