#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {512, 512};
  glm::vec2 kw = {4.f, 4.f};
  int       seed = 2;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN,
                                  shape,
                                  kw,
                                  seed,
                                  8,
                                  0.7f);
  z = hmap::bulkify(z, hmap::PrimitiveType::PRIM_CUBIC_PULSE, 1.f);

  hmap::remap(z);
  auto z0 = z;

  hmap::gpu::hydraulic_vpipes(z);

  z0.dump("z0.png");
  z.dump("z.png");


  hmap::export_banner_png("ex_hydraulic_vpipes.png",
                          {z0, z},
                          hmap::Cmap::TERRAIN, true);
}
