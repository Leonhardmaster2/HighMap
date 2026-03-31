#include <vector>

#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z);

  glm::ivec2 ij_start = {40, 40};
  glm::ivec2 ij_end = {230, 230};
  float      offset_ratio = 0.3f; // in [0, 1]

  // smallest resolution
  std::vector<glm::ivec2> ij1 = hmap::find_path_midpoint(z,
                                                         ij_start,
                                                         ij_end,
                                                         offset_ratio);

  // limit iterations
  int                     max_it = 4;
  std::vector<glm::ivec2> ij2 = hmap::find_path_midpoint(z,
                                                         ij_start,
                                                         ij_end,
                                                         offset_ratio,
                                                         max_it);

  // convert to a path for Spline interp
  hmap::Path path(ij2, shape);
  path.bspline();

  // --- export path to a png file

  hmap::Array w1 = hmap::Array(shape);
  for (const auto &p : ij1)
    w1(p) = 1.f;

  hmap::Array w2 = hmap::Array(shape);
  for (const auto &p : ij2)
    w2(p) = 1.f;

  hmap::Array w3 = hmap::Array(shape);
  path.to_array(w3);

  hmap::export_banner_png("ex_find_path_midpoint.png",
                          {z, w1, w2, w3},
                          hmap::Cmap::INFERNO);
}
