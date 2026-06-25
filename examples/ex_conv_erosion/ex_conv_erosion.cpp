#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 0;

  // shape = {1024, 1024};

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  // z0 = hmap::bulkify(z0, hmap::PrimitiveType::PRIM_CUBIC_PULSE, 1.f);
  z0 = hmap::bulkify(z0, hmap::PrimitiveType::PRIM_CONE_SMOOTH, 3.f);

  auto z1 = z0;

  int   iterations = 40;
  int   particle_count = 500;
  int   ir_min = 1;
  int   ir_max = 4;
  float size_distrib_exp = 1.f;
  float erosion_strength = 0.005f;
  float randomness = 0.02f;
  float exit_forcing = 0.05f;

  hmap::gpu::conv_erosion(z1,
                          seed,
                          iterations,
                          particle_count,
                          ir_min,
                          ir_max,
                          size_distrib_exp,
                          erosion_strength,
                          randomness,
                          exit_forcing);

  z1.dump("out.png");

  hmap::export_banner_png("ex_conv_erosion.png",
                          {z0, z1},
                          hmap::Cmap::TERRAIN,
                          true);
}
