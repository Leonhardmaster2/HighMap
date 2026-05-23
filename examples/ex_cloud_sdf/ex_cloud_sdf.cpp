#include "highmap.hpp"

int main(void)
{
  glm::ivec2    shape = {256, 256};
  std::uint32_t seed = 1;

  auto noise = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, {2, 2}, seed);
  hmap::remap(noise, 0.f, 0.2f);

  glm::vec4   bbox = {0.f, 1.f, 0.f, 1.f};
  hmap::Cloud cloud = hmap::Cloud(5, seed, hmap::adjust(bbox, -0.1f));

  hmap::Array z0 = cloud.to_array(shape, bbox);
  hmap::Array z1 = hmap::cloud_sdf_to_array(cloud, shape, bbox);
  hmap::Array z2 = hmap::cloud_sdf_to_array(cloud, shape, bbox, &noise, &noise);

  hmap::export_banner_png("ex_cloud_sdf.png", {z0, z1, z2}, hmap::Cmap::JET);
}
