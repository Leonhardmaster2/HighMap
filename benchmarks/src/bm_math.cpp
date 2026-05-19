#include "highmap/array.hpp"
#include "highmap/math/array.hpp"
#include "highmap/math/core.hpp"
#include "highmap/primitives.hpp"

#include <benchmark/benchmark.h>

using namespace hmap;

// ------------------------------------------------------------
// Args helper
// ------------------------------------------------------------

static void math_args(benchmark::internal::Benchmark *b)
{
  std::vector<int> sizes = {128, 256, 512, 1024};
  for (int n : sizes)
    b->Args({n});
}

// ------------------------------------------------------------
// ABS family
// ------------------------------------------------------------

static void BM_math_abs(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, -1.f, 1.f, 42);
    auto  out = abs(a);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_abs)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_math_abs_smooth(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, -1.f, 1.f, 42);
    auto  out = abs_smooth(a, 0.1f);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_abs_smooth)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ------------------------------------------------------------
// TRIG / EXP
// ------------------------------------------------------------

static void BM_math_sin(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 6.28f, 42);
    auto  out = sin(a);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_sin)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_math_cos(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 6.28f, 42);
    auto  out = cos(a);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_cos)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_math_exp(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, -2.f, 2.f, 42);
    auto  out = exp(a);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_exp)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ------------------------------------------------------------
// SQRT / LOG
// ------------------------------------------------------------
static void BM_math_sqrt(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 10.f, 42);
    auto  out = sqrt(a);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_sqrt)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_math_log10(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.1f, 10.f, 42);
    auto  out = log10(a);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_log10)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ------------------------------------------------------------
// POWER
// ------------------------------------------------------------

static void BM_math_pow(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 2.f, 42);
    auto  out = pow(a, 2.5f);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_pow)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ------------------------------------------------------------
// GAUSSIAN / SIGMOID
// ------------------------------------------------------------

static void BM_math_gaussian_decay(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, -2.f, 2.f, 42);
    auto  out = gaussian_decay(a, 1.f);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_gaussian_decay)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_math_sigmoid(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, -2.f, 2.f, 42);
    auto  out = sigmoid(a, 0.5f, 0.f, 1.f, 0.f);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_sigmoid)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ------------------------------------------------------------
// HYPOT / ATAN2
// ------------------------------------------------------------

static void BM_math_hypot(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, -1.f, 1.f, 42);
    Array b = white({n, n}, -1.f, 1.f, 43);
    auto  out = hypot(a, b);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_hypot)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_math_atan2(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, -1.f, 1.f, 42);
    Array b = white({n, n}, -1.f, 1.f, 43);
    auto  out = atan2(a, b);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_atan2)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ------------------------------------------------------------
// LERP
// ------------------------------------------------------------

static void BM_math_lerp_array(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    Array b = white({n, n}, 0.f, 1.f, 43);
    Array t = white({n, n}, 0.f, 1.f, 44);

    auto out = lerp(a, b, t);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_lerp_array)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ------------------------------------------------------------
// SMOOTHSTEP
// ------------------------------------------------------------

static void BM_math_smoothstep3(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    auto  out = smoothstep3(a, 0.2f, 0.8f);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_smoothstep3)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_math_smoothstep5(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    auto  out = smoothstep5(a, 0.2f, 0.8f);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_smoothstep5)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

// ------------------------------------------------------------
// THRESHOLD
// ------------------------------------------------------------

static void BM_math_threshold(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    auto  out = threshold(a, 0.3f, 0.7f);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_threshold)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_math_threshold_smooth(benchmark::State &state)
{
  int n = state.range(0);

  for (auto _ : state)
  {
    Array a = white({n, n}, 0.f, 1.f, 42);
    auto  out = threshold_smooth(a, 0.3f, 0.7f);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_math_threshold_smooth)
    ->Apply(math_args)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
