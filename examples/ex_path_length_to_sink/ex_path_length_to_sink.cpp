#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {2.f, 2.f};
  int        seed = 0;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);

  hmap::Array k_sink = hmap::path_length_to_sink(z);
  hmap::Array k_outlet = hmap::path_length_to_outlet(z);

  hmap::export_banner_png("ex_path_length_to_sink.png",
                          {z, k_sink, k_outlet},
                          hmap::Cmap::INFERNO,
                          false,
                          true);
}
