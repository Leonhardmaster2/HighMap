#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 0;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, ++seed);
  hmap::clamp_min(z, 0.f);
  hmap::remap(z);

  // mimic quantization
  z = hmap::quantize(z, 64);

  // try to improve
  float dither_amplitude = 0.01f;
  int   iterations = 4;
  auto  zd = hmap::dequantize(z, ++seed, dither_amplitude, iterations);

  hmap::export_banner_png("ex_dequantize.png",
                          {z, zd},
                          hmap::Cmap::INFERNO,
                          true);
}
