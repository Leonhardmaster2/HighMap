#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  // hmap::clamp_min(z, 0.f);
  hmap::remap(z);

  int ir = 16;

  hmap::Array marker_d = hmap::gpu::erosion(z, ir);
  hmap::Array zd = hmap::gpu::reconstruction_by_dilation(marker_d, z, ir);

  hmap::Array marker_e = hmap::gpu::dilation(z, ir);
  hmap::Array ze = hmap::gpu::reconstruction_by_erosion(marker_e, z, ir);

  hmap::Array zo = hmap::gpu::opening_by_reconstruction(z, ir);
  hmap::Array zc = hmap::gpu::closing_by_reconstruction(z, ir);

  hmap::export_banner_png("ex_reconstruction_by_dilation.png",
                          {z, zd, ze, zc, zo},
                          hmap::Cmap::TERRAIN,
                          true);
}
