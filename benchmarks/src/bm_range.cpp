#include "highmap/array.hpp"
#include "highmap/primitives.hpp"
#include "highmap/range.hpp"

#include <benchmark/benchmark.h>

using namespace hmap;

// ------------------------------------------------------------
// Args helper
// ------------------------------------------------------------

static void range_args(benchmark::internal::Benchmark *b)
{
  std::vector<int> sizes = {128, 256, 512, 1024};
  for (int n : sizes)
    b->Args({n});
}

// ------------------------------------------------------------
// Chop
// ------------------------------------------------------------

static void BM_range_chop(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    chop(a, 0.5f);
    benchmark::DoNotOptimize(a);
  }
}
BENCHMARK(BM_range_chop)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_range_chop_max_smooth(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    chop_max_smooth(a, 0.8f);
    benchmark::DoNotOptimize(a);
  }
}
BENCHMARK(BM_range_chop_max_smooth)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ------------------------------------------------------------
// Clamp family
// ------------------------------------------------------------

static void BM_range_clamp(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, -1.f, 1.f, 42);
    clamp(a, -0.5f, 0.5f);
    benchmark::DoNotOptimize(a);
  }
}
BENCHMARK(BM_range_clamp)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_range_clamp_mode(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, -1.f, 1.f, 42);
    clamp(a, 0.5f, ClampMode::BOTH);
    benchmark::DoNotOptimize(a);
  }
}
BENCHMARK(BM_range_clamp_mode)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_range_clamp_max(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    clamp_max(a, 0.7f);
    benchmark::DoNotOptimize(a);
  }
}
BENCHMARK(BM_range_clamp_max)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_range_clamp_max_array(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    Array b = white({n, n}, 0.3f, 0.9f, 1337);
    clamp_max(a, b);
    benchmark::DoNotOptimize(a);
  }
}
BENCHMARK(BM_range_clamp_max_array)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_range_clamp_max_smooth(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    clamp_max_smooth(a, 0.7f, 0.1f);
    benchmark::DoNotOptimize(a);
  }
}
BENCHMARK(BM_range_clamp_max_smooth)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_range_clamp_min(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    clamp_min(a, 0.3f);
    benchmark::DoNotOptimize(a);
  }
}
BENCHMARK(BM_range_clamp_min)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_range_clamp_min_array(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    Array b = white({n, n}, 0.1f, 0.5f, 1337);
    clamp_min(a, b);
    benchmark::DoNotOptimize(a);
  }
}
BENCHMARK(BM_range_clamp_min_array)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_range_clamp_min_smooth(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    clamp_min_smooth(a, 0.3f, 0.1f);
    benchmark::DoNotOptimize(a);
  }
}
BENCHMARK(BM_range_clamp_min_smooth)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_range_clamp_smooth(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    clamp_smooth(a, 0.2f, 0.8f, 0.1f);
    benchmark::DoNotOptimize(a);
  }
}
BENCHMARK(BM_range_clamp_smooth)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ------------------------------------------------------------
// Max / Min
// ------------------------------------------------------------

static void BM_range_maximum(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    Array b = white({n, n}, 0.f, 1.f, 1337);
    Array out = maximum(a, b);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_range_maximum)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_range_maximum_smooth(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    Array b = white({n, n}, 0.f, 1.f, 1337);
    Array out = maximum_smooth(a, b, 0.1f);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_range_maximum_smooth)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_range_minimum(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    Array b = white({n, n}, 0.f, 1.f, 1337);
    Array out = minimum(a, b);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_range_minimum)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_range_minimum_smooth(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    Array b = white({n, n}, 0.f, 1.f, 1337);
    Array out = minimum_smooth(a, b, 0.1f);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_range_minimum_smooth)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ------------------------------------------------------------
// Remap / Rescale
// ------------------------------------------------------------

static void BM_range_remap_auto(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    remap(a, 0.f, 1.f);
    benchmark::DoNotOptimize(a);
  }
}
BENCHMARK(BM_range_remap_auto)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_range_remap_explicit(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    remap(a, 0.f, 1.f, 0.f, 1.f);
    benchmark::DoNotOptimize(a);
  }
}
BENCHMARK(BM_range_remap_explicit)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_range_rescale(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    rescale(a, 2.0f, 0.5f);
    benchmark::DoNotOptimize(a);
  }
}
BENCHMARK(BM_range_rescale)
    ->Apply(range_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
