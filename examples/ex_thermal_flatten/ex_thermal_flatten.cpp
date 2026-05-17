#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);

  hmap::Array talus = hmap::Array(shape, 1.5f / shape.x);
  int         iterations = 500;
  float       sigma_inf = 0.5f;
  float       sigma_sup = 0.f;

  auto z1 = z0;
  hmap::gpu::thermal_flatten(z1, talus, iterations, sigma_inf, sigma_sup);

  auto z2 = z0;
  sigma_inf = 0.5f;
  sigma_sup = 0.05f;

  hmap::gpu::thermal_flatten(z2, talus, iterations, sigma_inf, sigma_sup);

  hmap::export_banner_png("ex_thermal_flatten.png",
                          {z0, z1, z2},
                          hmap::Cmap::TERRAIN,
                          true);
}
