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

  int iterations = 16;

  auto sk0 = hmap::skeleton(z);
  auto sk1 = hmap::remove_endpoints(sk0, iterations);

  auto overlay = hmap::maximum(0.2f * sk0, sk1);

  hmap::export_banner_png("ex_remove_endpoints.png",
                          {sk0, sk1, overlay},
                          hmap::Cmap::GRAY);
}
