#include <benchmark/benchmark.h>

#include "highmap/array.hpp"
#include "highmap/curvature.hpp"
#include "highmap/primitives.hpp"

using namespace hmap;

static void BM_curvature_quadric_CPU(benchmark::State &state)
{
  const int           n = state.range(0); // mesh size
  const int           r = state.range(1); // kernel radius
  const CurvatureType type = static_cast<CurvatureType>(state.range(2));

  Array input = white(glm::vec2(n, n), 0.f, 1.f, 42);

  for (auto _ : state)
  {
    Array out = input;
    hmap::curvature_quadric(out, r, type);
    benchmark::DoNotOptimize(out);
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * n * n);
}

static void BM_curvature_quadric_GPU(benchmark::State &state)
{
  const int           n = state.range(0); // mesh size
  const int           r = state.range(1); // kernel radius
  const CurvatureType type = static_cast<CurvatureType>(state.range(2));

  Array input = white(glm::vec2(n, n), 0.f, 1.f, 42);

  for (auto _ : state)
  {
    Array out = input;
    hmap::gpu::curvature_quadric(out, r, type);
    benchmark::DoNotOptimize(out);
  }

  state.SetItemsProcessed(int64_t(state.iterations()) * n * n);
}

static void curvature_quadric_args(benchmark::internal::Benchmark *b)
{
  // remove GPU-only available curvatures
  std::vector<int> sizes = {128, 256, 512, 1024};
  std::vector<int> radii = {1, 4, 8, 16, 32};
  std::vector<int> types = {
      (int)CurvatureType::CT_MIN,
      (int)CurvatureType::CT_MAX,
      (int)CurvatureType::CT_MEAN,
      (int)CurvatureType::CT_GAUSSIAN,
      (int)CurvatureType::CT_PROFILE,
      (int)CurvatureType::CT_CONTOUR,
      (int)CurvatureType::CT_TANGENTIAL,
      // (int)CurvatureType::CT_ACCUMULATION,
      // (int)CurvatureType::CT_SHAPE_INDEX,
      // (int)CurvatureType::CT_UNSPHERICITY,
      // (int)CurvatureType::CT_RING,
      // (int)CurvatureType::CT_ROTOR
  };

  for (int n : sizes)
    for (int r : radii)
      for (int t : types)
        b->Args({n, r, t});
}

BENCHMARK(BM_curvature_quadric_CPU)->Apply(curvature_quadric_args);
BENCHMARK(BM_curvature_quadric_GPU)->Apply(curvature_quadric_args);
