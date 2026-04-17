#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  uint       seed = 1;

  glm::vec4  bbox = {0.2f, 0.8f, 0.2f, 0.8f};
  hmap::Path path = hmap::Path(5, seed, bbox);
  path.reorder_nns();
  path.print();

  glm::vec4 bbox_array = {0.f, 1.f, 0.f, 1.f};

  hmap::Array z_sdf_o = path.to_array_sdf(shape, bbox_array);

  path.set_closed(true);
  hmap::Array z_sdf_c = path.to_array_sdf(shape, bbox_array);

  hmap::export_banner_png("ex_path_sdf.png",
                          {z_sdf_o, z_sdf_c},
                          hmap::Cmap::JET);
}
