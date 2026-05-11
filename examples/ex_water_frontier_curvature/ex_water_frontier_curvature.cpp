#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::remap(z);

  hmap::Array water_depth = hmap::flooding_uniform_level(z, 0.5f);

  int prefilter_ir = 16;

  hmap::Array wc0 = hmap::water_frontier_curvature(water_depth, prefilter_ir);
  hmap::Array wc1 = hmap::gpu::water_frontier_curvature(water_depth,
                                                        prefilter_ir);

  hmap::export_banner_png("ex_water_frontier_curvature.png",
                          {z, water_depth, wc0, wc1},
                          hmap::Cmap::JET,
                          false,
                          true);
}
