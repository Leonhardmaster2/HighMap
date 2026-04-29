#include <benchmark/benchmark.h>

#include "highmap/array.hpp"
#include "highmap/filters.hpp"

using namespace hmap;

static void BM_smooth_cpulse_CPU(benchmark::State &state)
{
  const int n = state.range(0); // mesh size
  const int r = state.range(1); // kernel radius

  Array input = white(glm::vec2(n, n), 0.f, 1.f, 42);

  for (auto _ : state)
  {
    Array out = input;
    hmap::smooth_cpulse(out, r);
    benchmark::DoNotOptimize(out);
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * n * n);
}

static void BM_smooth_cpulse_GPU(benchmark::State &state)
{
  const int n = state.range(0); // mesh size
  const int r = state.range(1); // kernel radius

  Array input = white(glm::vec2(n, n), 0.f, 1.f, 42);

  for (auto _ : state)
  {
    Array out = input;
    hmap::gpu::smooth_cpulse(out, r);
    benchmark::DoNotOptimize(out);
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * n * n);
}

static void smooth_cpulse_args(benchmark::internal::Benchmark *b)
{
  std::vector<int> sizes = {128, 256, 512, 1024};
  std::vector<int> radii = {1, 4, 8, 16, 32};

  for (int n : sizes)
    for (int r : radii)
      b->Args({n, r});
}

BENCHMARK(BM_smooth_cpulse_CPU)->Apply(smooth_cpulse_args);
BENCHMARK(BM_smooth_cpulse_GPU)->Apply(smooth_cpulse_args);
