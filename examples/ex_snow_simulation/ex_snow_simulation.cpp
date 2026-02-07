#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  // shape = {512, 512};
  glm::vec2 kw = {4.f, 4.f};
  int       seed = 2;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  z = hmap::bulkify(z, hmap::PrimitiveType::PRIM_CUBIC_PULSE, 2.f);
  hmap::remap(z);

  hmap::Array talus(shape, 2.5f / shape.x);

  hmap::Array melt = smoothstep5(z, 0.f, 0.4f);
  hmap::remap(melt, 1.f, 0.f);

  hmap::Array snow_depth = hmap::gpu::snow_simulation(z,
                                                      1e-1f,
                                                      hmap::Array(shape, 1.f),
                                                      melt,
                                                      talus,
                                                      200);

  z.dump("z0.png");
  (z + snow_depth).dump("z.png");
  snow_depth.dump("depth.png");

  hmap::export_banner_png("ex_snow_simulation.png",
                          {z, z + snow_depth, snow_depth},
                          hmap::Cmap::JET);
}
