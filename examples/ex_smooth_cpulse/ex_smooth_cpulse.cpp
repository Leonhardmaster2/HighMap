#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 1;
  int        ir = 64;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::Array cpu = z0;
  hmap::Array gpu = z0;

  hmap::smooth_cpulse(cpu, ir);
  hmap::gpu::smooth_cpulse(gpu, ir);

  hmap::export_banner_png("ex_smooth_cpulse.png",
                          {z0, cpu, gpu},
                          hmap::Cmap::JET,
                          false,
                          true);
}
