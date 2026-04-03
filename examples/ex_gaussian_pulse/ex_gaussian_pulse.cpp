#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  float      sigma = 0.2f;

  hmap::Array z1 = hmap::gaussian_pulse(shape, sigma);

  // with control array
  glm::vec2   kw = {4.f, 4.f};
  int         seed = 1;
  hmap::Array ctrl_array = hmap::noise(hmap::NoiseType::PERLIN,
                                       shape,
                                       kw,
                                       seed);
  hmap::remap(ctrl_array, 0.5f, 1.5f);

  hmap::Array z2 = hmap::gaussian_pulse(shape, sigma, &ctrl_array);

  hmap::export_banner_png("ex_gaussian_pulse.png",
                          {z1, z2},
                          hmap::Cmap::INFERNO,
                          false);
}
