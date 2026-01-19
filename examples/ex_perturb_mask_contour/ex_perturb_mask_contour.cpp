#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {8.f, 8.f};
  int        seed = 1;

  hmap::Array mask = hmap::cone(shape, 3.f);
  hmap::make_binary(mask, 0.f);

  hmap::Array noise = hmap::noise_fbm(hmap::NoiseType::PERLIN,
                                      shape,
                                      res,
                                      seed);

  float max_displacement = 32.f; // pixels
  auto  mask_p = hmap::perturb_mask_contour(mask, noise, max_displacement);

  hmap::export_banner_png("ex_perturb_mask_contour.png",
                          {mask, noise, mask_p},
                          hmap::Cmap::GRAY);
}
