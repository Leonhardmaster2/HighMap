#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};

  auto z1 = hmap::polar_shape(shape);

  float rmin = 0.3f;
  float rmax = 0.4f;
  float aspect_ratio = 2.f;
  float smoothing_width = 0.1f;
  bool  square_base = true;

  auto z2 = hmap::polar_shape(shape,
                              rmin,
                              rmax,
                              aspect_ratio,
                              smoothing_width,
                              square_base);

  hmap::export_banner_png("ex_polar_shape.png", {z1, z2}, hmap::Cmap::MAGMA);
}
