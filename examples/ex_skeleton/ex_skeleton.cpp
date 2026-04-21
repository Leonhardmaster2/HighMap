#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {2.f, 8.f};
  int        seed = 2;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::remap(z, -1.f, 0.4f);

  hmap::make_binary(z);

  auto sk = hmap::skeleton(z);
  auto skg = hmap::gpu::skeleton(z);

  int  ir_search = 32;
  auto rdist = relative_distance_from_skeleton(z, ir_search);

  hmap::export_banner_png("ex_skeleton.png",
                          {z, sk, skg, 0.5f * (z + sk), rdist},
                          hmap::Cmap::GRAY);
}
