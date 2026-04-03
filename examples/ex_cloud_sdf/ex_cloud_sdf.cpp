#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  uint       seed = 1;

  auto noise = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, {2, 2}, seed);
  hmap::remap(noise, 0.f, 0.2f);

  glm::vec4   bbox = {0.2f, 0.8f, 0.2f, 0.8f};
  hmap::Cloud cloud = hmap::Cloud(5, seed, bbox);

  glm::vec4 bbox_array = {0.f, 1.f, 0.f, 1.f};

  hmap::Array z_sdf1 = cloud.to_array_sdf(shape, bbox_array);
  hmap::Array z_sdf2 = cloud.to_array_sdf(shape, bbox_array, &noise, &noise);

  hmap::export_banner_png("ex_cloud_sdf.png",
                          {z_sdf1, z_sdf2},
                          hmap::Cmap::JET);
}
