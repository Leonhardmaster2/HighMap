#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {8.f, 8.f};
  int        seed = 1;

  // generate a mask with "staircaise" borders
  hmap::Array mask = hmap::noise_fbm(hmap::NoiseType::PERLIN,
                                     {32, 32},
                                     res,
                                     seed);
  hmap::clamp_min(mask, 0.f);
  hmap::make_binary(mask);
  mask = mask.resample_to_shape_nearest(shape);

  // apply mask to field
  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z);
  z *= mask;

  // smooth contour
  int         ir = 8;
  float       transition_ratio = 0.2f;
  hmap::Array contour = hmap::contour_smoothing(z, ir, transition_ratio);

  hmap::export_banner_png("ex_contour_smoothing.png",
                          {mask, z, contour, is_non_zero(contour), z * contour},
                          hmap::Cmap::JET);
}
