#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  int seed = 6;
  int npoints = 5;

  glm::vec4  bbox = {1.f, 2.f, -0.5f, 0.5f};
  hmap::Path path = hmap::Path(npoints, seed, {1.3f, 1.7f, -0.2, 0.2f});
  path.reorder_nns();
  path.fractalize(/* iterations */ 2, seed, /* sigma */ 0.2f);
  // path.bspline();

  auto z1 = hmap::Array(shape);
  path.to_array(z1, bbox);

  // display curvature
  hmap::Path path_cpy = path;
  path_cpy.bspline();

  auto c = path_cpy.get_curvature(false);
  
  std::vector<glm::vec2> normals = path_cpy.get_normals();

  float dr = 0.001f;

  for (size_t k = 0; k < path_cpy.get_npoints(); ++k)
  {
    float amp = std::clamp(dr * c[k], -0.1f, 0.1f);
    path_cpy.points[k].x -= amp * normals[k].x;
    path_cpy.points[k].y -= amp * normals[k].y;
  }

  auto z2 = hmap::Array(shape);
  path.to_array(z2, bbox);
  path_cpy.to_array(z2, bbox);

  hmap::export_banner_png("ex_path_curvature.png",
                          {z1, z2},
                          hmap::Cmap::INFERNO);
}
