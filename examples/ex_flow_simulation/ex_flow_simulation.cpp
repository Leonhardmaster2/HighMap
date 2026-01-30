#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 2;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN,
                                  shape,
                                  kw,
                                  seed,
                                  8,
                                  0.7f);
  z = hmap::bulkify(z, hmap::PrimitiveType::PRIM_CUBIC_PULSE, -1.f);
  hmap::remap(z);

  hmap::Array water_depth = hmap::gpu::flow_simulation(z,
                                                       1e-2f,
                                                       hmap::Array(shape, 1.f),
                                                       200);

  z.dump("z.png");
  water_depth.dump("depth.png");

  hmap::export_banner_png("ex_flow_simulation.png",
                          {z, z + water_depth, water_depth},
                          hmap::Cmap::JET);
}
