#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 0;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z);

  hmap::Array count(shape);
  int         nsamples = 1000;

  for (int it = 0; it < nsamples; ++it)
  {
    for (const auto &b : {hmap::DomainBoundary::BOUNDARY_LEFT,
                          hmap::DomainBoundary::BOUNDARY_RIGHT,
                          hmap::DomainBoundary::BOUNDARY_TOP,
                          hmap::DomainBoundary::BOUNDARY_BOTTOM})
    {
      glm::ivec2 p = hmap::pick_boundary_cell(z, b, ++seed);
      count(p) += 1;
    }
  }

  count = hmap::dilation(count, 8); // to be able to see something
  hmap::remap(count);

  hmap::export_banner_png("ex_pick_boundary_cell.png",
                          {z, count},
                          hmap::Cmap::JET);
}
