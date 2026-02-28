#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {2.f, 2.f};
  int        seed = 0;

  auto dx = 0.2f *
            hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed, 8, 0.f);
  auto z0 = step(shape, 0.f, 3.f, nullptr, &dx);

  auto z1 = z0;

  float depth = 0.2f;
  int   iterations = 2500;

  hmap::gpu::mudslide(z1, 1.f / shape.x, depth, iterations);

  auto mask = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed + 1);
  make_binary(mask, 0.f);

  auto z2 = z0;
  hmap::gpu::mudslide(z2, mask, depth, iterations);

  hmap::export_banner_png("ex_mudslide.png",
                          {z0, z1, mask, z2},
                          hmap::Cmap::TERRAIN,
                          true);
}
