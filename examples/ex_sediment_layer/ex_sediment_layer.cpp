#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  z0 = hmap::bulkify(z0, hmap::PrimitiveType::PRIM_CUBIC_PULSE, 1.f);

  float talus_layer = 0.5f / shape.x;
  float talus_depo = 5.f / shape.x;

  // exclusion limit
  int  iterations = 500;
  bool apply_post_filter = true;

  auto z1 = z0;
  hmap::gpu::sediment_layer(z1,
                            hmap::Array(shape, talus_layer),
                            hmap::Array(shape, talus_depo),
                            iterations,
                            apply_post_filter);

  z1.to_png_grayscale("out.png", CV_16U);

  hmap::export_banner_png("ex_sediment_layer.png",
                          {z0, z1},
                          hmap::Cmap::TERRAIN,
                          true);
}
