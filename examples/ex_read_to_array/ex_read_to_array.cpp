#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 128};
  glm::vec2  res = {4.f, 2.f};
  int        seed = 1;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z0, 0.5f, 1.5f);

  z0.to_png("out1.png", hmap::Cmap::GRAY); // grayscale
  z0.to_png("out2.png", hmap::Cmap::JET);  // RGB image
  z0.to_exr("out.exr"); // preserve absolute values range [0.5, 1.5]

  hmap::Array z1 = hmap::read_to_array("out1.png");

  // converted to grayscale, will be different from input field z0
  hmap::Array z2 = hmap::read_to_array("out2.png");

  // keep range
  bool        flip_j = false;
  bool        remap = false;
  hmap::Array ze = hmap::read_to_array("out.exr", flip_j, remap);

  ze.infos("ze"); // check for range [0.5, 1.5]

  hmap::Array z3 = hmap::Array("out1.png");
  hmap::Array z4 = hmap::Array("out.exr"); // auto, in [0.5, 1.5

  hmap::export_banner_png("ex_read_to_array.png",
                          {z0, z1, z2, z3, ze, z4},
                          hmap::Cmap::JET);
}
