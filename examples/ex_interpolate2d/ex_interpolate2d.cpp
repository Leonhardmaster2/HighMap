#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  int        seed = 0;

  glm::vec4 bbox = {-1.f, 0.f, 0.5f, 1.5f};

  hmap::Cloud cloud = hmap::Cloud(20, seed, bbox);

  std::vector<float> x = cloud.get_x();
  std::vector<float> y = cloud.get_y();
  std::vector<float> values = cloud.get_values();

  // reference, pointwise values
  hmap::Array z0 = hmap::Array(shape);
  cloud.to_array(z0, bbox);

  // nearest
  hmap::Array z1 = hmap::interpolate2d(
      shape,
      x,
      y,
      values,
      hmap::InterpolationMethod2D::ITP2D_NEAREST,
      nullptr,
      nullptr,
      nullptr,
      bbox);

  hmap::Array z2 = hmap::interpolate2d(
      shape,
      x,
      y,
      values,
      hmap::InterpolationMethod2D::ITP2D_DELAUNAY,
      nullptr,
      nullptr,
      nullptr,
      bbox);

  hmap::Array z3 = hmap::interpolate2d(shape,
                                       x,
                                       y,
                                       values,
                                       hmap::InterpolationMethod2D::ITP2D_IDW,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       bbox);

  hmap::Array z4 = hmap::interpolate2d(
      shape,
      x,
      y,
      values,
      hmap::InterpolationMethod2D::ITP2D_GAUSSIAN,
      nullptr,
      nullptr,
      nullptr,
      bbox);

  hmap::export_banner_png("ex_interpolate2d.png",
                          {z0, z1, z2, z3, z4},
                          hmap::Cmap::INFERNO);
}
