#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {2.f, 2.f};
  int        seed = 0;

  hmap::Array z1 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::remap(z1);

  float gamma = 0.5f;
  float kz = 8.f; // 8-layers
  bool  linear_gamma = false;
  float gamma_noise_ratio = 0.1f;

  hmap::Array z2 = z1;
  hmap::gpu::strata_terrace(z2,
                            gamma,
                            seed,
                            kz,
                            linear_gamma,
                            gamma_noise_ratio);

  hmap::Array z3 = z1;
  hmap::Array noise = 0.1f * hmap::noise_fbm(hmap::NoiseType::PERLIN,
                                             shape,
                                             {16.f, 16.f},
                                             seed + 1);

  hmap::gpu::strata_terrace(z3,
                            gamma,
                            seed,
                            kz,
                            linear_gamma,
                            gamma_noise_ratio,
                            &noise);

  hmap::export_banner_png("ex_strata_terrace.png",
                          {z1, z2, z3},
                          hmap::Cmap::TERRAIN,
                          true);
}
