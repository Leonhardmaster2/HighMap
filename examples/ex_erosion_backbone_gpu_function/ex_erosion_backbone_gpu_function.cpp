#include "macrologger.h"

#include "highmap.hpp"
#include "highmap/dbg/timer.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  // --- input heighmap

  glm::ivec2 shape = {512, 512};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 0;
  float      bulk = 0.5f; // add some central bulk to facilitate outflow

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  z0 = hmap::bulkify(z0, hmap::PrimitiveType::PRIM_CONE_SMOOTH, bulk);

  // --- erosion (GPU with buffer)

  hmap::Array z1 = z0;
  float       some_parameter = 0.f;

  // applied in-place
  hmap::gpu::erosion_backbone_gpu_function_buffer(z1, some_parameter);

  // --- erosion (GPU with image sampler)

  hmap::Array z2 = z0;
  hmap::gpu::erosion_backbone_gpu_function_image(z2, some_parameter);

  // --- export

  // indivudal exports (opng, grayscale, 16bit)
  z0.dump("z0.png");
  z1.dump("z1.png");
  z2.dump("z2.png");

  // visual check
  hmap::export_banner_png("ex_erosion_backbone_gpu_function.png",
                          {z0, z1, z2},
                          hmap::Cmap::TERRAIN,
                          true);
}
