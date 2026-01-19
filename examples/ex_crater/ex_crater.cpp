#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 1;

  // without noise
  float radius = 0.2f;
  float lip_decay = 0.12f;
  float depth = 0.5f;

  hmap::Array z1 = hmap::crater(shape, radius, depth, lip_decay);

  // with control array
  hmap::Array ctrl_array = hmap::noise(hmap::NoiseType::PERLIN,
                                       shape,
                                       kw,
                                       seed++);
  hmap::remap(ctrl_array);

  float lip_height_ratio = 0.5f;

  hmap::Array z2 = hmap::crater(shape,
                                radius,
                                depth,
                                lip_decay,
                                lip_height_ratio,
                                &ctrl_array);

  hmap::export_banner_png("ex_crater.png", {z1, z2}, hmap::Cmap::TERRAIN, true);
  z1.to_file("out.bin");
}
