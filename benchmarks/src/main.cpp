#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include "highmap/opencl/gpu_opencl.hpp"

void global_init()
{
  hmap::gpu::init_opencl();
  spdlog::set_level(spdlog::level::off);
}

int main(int argc, char** argv)
{
  global_init();

  benchmark::Initialize(&argc, argv);

  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;

  benchmark::RunSpecifiedBenchmarks();

  return 0;
}
