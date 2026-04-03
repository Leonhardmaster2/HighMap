#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);

  glm::ivec2 new_shape = {1024, 1024};

  hmap::Array z1 = z0.resample_to_shape(new_shape);
  z1 = z1.resample_to_shape(shape);

  hmap::Array z2 = z0.resample_to_shape_bicubic(new_shape);
  z2.to_png("out.png", hmap::Cmap::JET);
  z2 = z2.resample_to_shape(shape);

  hmap::export_banner_png("ex_array_interp.png",
                          {z0, z1, z2},
                          hmap::Cmap::MAGMA);
}
