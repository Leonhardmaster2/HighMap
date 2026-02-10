#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {2.f, 2.f};
  int        seed = 0;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::SIMPLEX2, shape, kw, seed);
  hmap::remap(z);

  // --- CPU phase field

  float kp = 4.f;
  int   width = 64;

  float density = 32.f;

  hmap::Array phi0 = hmap::phase_field(z, kp, width, ++seed, -1, density);

  bool        rotate90 = true;
  hmap::Array phi1 =
      hmap::phase_field(z, kp, width, ++seed, -1, density, rotate90);

  hmap::remap(phi0);
  hmap::remap(phi1);

  // --- GPU phase field

  glm::vec2   kcells = {8.f, 8.f};
  hmap::Array phi2 = hmap::gpu::phase_field(z, kcells, seed);
  hmap::remap(phi2);

  hmap::export_banner_png("ex_phase_field.png",
                          {z, phi0, phi1, phi2},
                          hmap::Cmap::JET);
}
