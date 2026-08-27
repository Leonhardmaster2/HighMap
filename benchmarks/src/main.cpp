#include "highmap/opencl/gpu_opencl.hpp"

#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

void global_init()
{
  // OpenCL may be unavailable even when its framework is installed. OpenCL
  // benchmark cases report that condition themselves instead of aborting the
  // complete CPU/Metal benchmark executable.
  (void)hmap::gpu::init_opencl();
  spdlog::set_level(spdlog::level::off);
}

int main(int argc, char **argv)
{
  global_init();

  benchmark::Initialize(&argc, argv);

  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

  benchmark::RunSpecifiedBenchmarks();

  return 0;
}
