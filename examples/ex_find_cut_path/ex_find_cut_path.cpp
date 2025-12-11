#include "highmap.hpp"

int main(void)
{
  hmap::Vec2<int>   shape = {256, 256};
  hmap::Vec2<float> kw = {2.f, 2.f};
  int               seed = 0;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::remap(z);

  hmap::Path path = hmap::find_cut_path(z,
                                        hmap::DomainBoundary::BOUNDARY_LEFT,
                                        hmap::DomainBoundary::BOUNDARY_RIGHT);

  hmap::Array zp = hmap::Array(shape);
  path.to_array(zp);

  hmap::export_banner_png("ex_find_cut_path.png", {z, zp}, hmap::Cmap::INFERNO);
}
