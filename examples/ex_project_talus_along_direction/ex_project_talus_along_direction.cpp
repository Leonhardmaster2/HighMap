#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 0;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);

  std::vector<hmap::Array> z_list = {z};

  hmap::Array talus(shape, 2.f / shape.x);

  for (int direction = 0; direction < 8; ++direction)
  {
    // clang-format off
    // 'D8' hydrology convention
    // 5 6 7
    // 4 . 0
    // 3 2 1
    // clang-format on

    hmap::Array zp = hmap::gpu::project_talus_along_direction(z,
                                                              talus,
                                                              direction);

    if (direction == 0) zp.dump();

    z_list.push_back(zp);
  }

  hmap::export_banner_png("ex_project_talus_along_direction.png",
                          z_list,
                          hmap::Cmap::JET);
}
