#include <iostream>

#include "highmap.hpp"
#include "highmap/dbg/timer.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2    shape = {256, 256};
  glm::vec2     kw = {2.f, 2.f};
  std::uint32_t seed = 0;

  hmap::Array z0 = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);

  hmap::Array mask_fixed_values = z0;
  hmap::clamp_min(mask_fixed_values, 0.f);

  mask_fixed_values.infos();

  hmap::Timer::Start("CPU SOR");
  hmap::Array z1 = hmap::harmonic_interpolation(z0, mask_fixed_values);
  hmap::Timer::Stop("CPU SOR");

  hmap::Timer::Start("GPU (Legacy BF)");
  hmap::Array z_legacy = hmap::gpu::harmonic_interpolation_legacy_bf(
      z0,
      mask_fixed_values);
  hmap::Timer::Stop("GPU (Legacy BF)");

  hmap::Timer::Start("GPU (Red-Black SOR)");
  hmap::Array z2 = hmap::gpu::harmonic_interpolation(z0, mask_fixed_values);
  hmap::Timer::Stop("GPU (Red-Black SOR)");

  hmap::export_banner_png("ex_harmonic_interpolation.png",
                          {z0, mask_fixed_values, z1, z2},
                          hmap::Cmap::JET,
                          false);
}
