#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  // shape = {1024, 1024};
  glm::vec2 kw = {4.f, 4.f};
  int       seed = 0;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN,
                                   shape,
                                   kw,
                                   seed,
                                   8,
                                   0.f);
  z0 = hmap::bulkify(z0, hmap::PrimitiveType::PRIM_CONE_SMOOTH, 1.f);
  hmap::remap(z0);

  int nparticles = int(0.9f * shape.x * shape.y);

  auto z1 = z0;
  auto zb = 0.8f * z0; // bedrock

  hmap::gpu::hydraulic_particle(z1, nparticles, seed, &zb);

  // --- output

  z0.dump("out0.png");
  z1.dump("out1.png");

  hmap::export_banner_png("ex_hydraulic_particle.png",
                          {z0, z1},
                          hmap::Cmap::TERRAIN,
                          true);
}
