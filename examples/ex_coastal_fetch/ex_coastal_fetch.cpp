#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  z = hmap::bulkify(z, hmap::PrimitiveType::PRIM_CONE_SMOOTH, 1.f);

  int ndirections = 128;

  // --- base isotropic

  hmap::Array f1 = hmap::gpu::coastal_fetch(z, ndirections);

  // with compute mask (compute only where mask is positive)
  auto cmask = z;
  hmap::remap(cmask, -0.5, 0.5f);

  hmap::Array f2 = hmap::gpu::coastal_fetch(z, ndirections, &cmask);

  // --- directional

  float angle = 15.f; // degs
  float exp = 1.f;    // increase to sharpen selection

  hmap::Array f3 = hmap::gpu::coastal_fetch_directional(z,
                                                        angle,
                                                        exp,
                                                        ndirections);
  hmap::Array f4 = hmap::gpu::coastal_fetch_directional(z,
                                                        angle,
                                                        4.f * exp,
                                                        ndirections);

  // --- output (normalized)

  hmap::export_banner_png("ex_coastal_fetch.png",
                          {z, f1, f2, f3, f4},
                          hmap::Cmap::JET,
                          false,
                          true);
}
