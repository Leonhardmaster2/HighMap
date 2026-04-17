#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  uint       seed = 1;

  glm::vec4  bbox = {0.f, 1.f, 0.f, 1.f};
  hmap::Path path = hmap::Path(15, seed, hmap::adjust(bbox, -0.1f));
  path.reorder_nns();

  auto z0 = path.to_array(shape, bbox);
  auto z1 = hmap::path_sdf_to_array(path, shape, bbox);

  path.set_closed(true);
  auto z2 = hmap::path_sdf_to_array(path, shape, bbox);

  hmap::export_banner_png("ex_path_sdf.png", {z0, z1, z2}, hmap::Cmap::INFERNO);
}
