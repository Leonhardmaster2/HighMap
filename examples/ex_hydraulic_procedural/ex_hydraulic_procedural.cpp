#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  // shape = {1024, 1024};
  glm::vec2 kw = {2.f, 2.f};
  int       seed = 0;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::SIMPLEX2,
                                   shape,
                                   kw,
                                   seed,
                                   8);
  // z0 = hmap::bulkify(z0, hmap::PrimitiveType::PRIM_CONE_SMOOTH, 1.f);
  hmap::remap(z0);

  auto  z1 = z0;
  float kp_global = 24.f;
  float c_erosion = 0.2f;
  auto  erosion_profile = hmap::ErosionProfile::EP_TRIANGLE_GRENIER;

  hmap::gpu::hydraulic_procedural_fbm(z1,
                                      kp_global,
                                      c_erosion,
                                      seed,
                                      erosion_profile);

  auto z2 = z0;
  hmap::gpu::hydraulic_procedural(z2, kp_global, c_erosion, seed);

  z0.dump("out0.png");
  z1.dump("out1.png");
  z2.dump("out2.png");

  hmap::export_banner_png("ex_hydraulic_procedural0.png",
                          {z0, z1, z2},
                          hmap::Cmap::TERRAIN,
                          true);

  return 0;

  // --- all profiles

  auto profiles = hmap::check_erosion_profile_function();

  std::vector<hmap::Array> stack = {z0};
  for (auto ep : profiles)
  {
    hmap::Array ze = z0;
    hmap::gpu::hydraulic_procedural(ze, kp_global, c_erosion, seed, ep);
    stack.push_back(ze);
  }

  hmap::export_banner_png("ex_hydraulic_procedural1.png",
                          stack,
                          hmap::Cmap::TERRAIN,
                          true);
}
