#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  z0 = hmap::bulkify(z0, hmap::PrimitiveType::PRIM_CONE_SMOOTH, 1.f);
  hmap::remap(z0);

  hmap::Array z1 = z0;

  float     amp = 0.5f;
  glm::vec2 kc = {4.f, 8.f};
  float     gamma = 0.4f;
  float     gamma_lateral = 0.4f;
  float     angle = 5.f;
  bool      enable_default_noise = true;
  float     noise_amp = 0.1f;
  bool      absolute_displacement = false;
  float     occurence_probability = 1.f;

  hmap::gpu::strata_cells(z1,
                          kc,
                          amp,
                          seed,
                          gamma,
                          gamma_lateral,
                          angle,
                          noise_amp,
                          absolute_displacement,
                          occurence_probability);
  hmap::remap(z1);

  int   octaves = 6;
  float persistence = 0.4f;
  float lacunarity = 2.2f;

  auto z2 = z0;
  float default_noise_amp = 0.02f;
  
  hmap::gpu::strata_cells_fbm(z2,
                              kc,
                              amp,
                              seed,
                              gamma,
                              gamma_lateral,
                              angle,
                              enable_default_noise,
                              default_noise_amp,
                              absolute_displacement,
                              occurence_probability,
                              octaves,
                              persistence,
                              lacunarity);
  hmap::remap(z2);

  z0.dump("out0.png");
  z1.dump("out1.png");
  z2.dump("out2.png");

  hmap::export_banner_png("ex_strata_cells.png",
                          {z0, z1, z2},
                          hmap::Cmap::TERRAIN,
                          true);
}
