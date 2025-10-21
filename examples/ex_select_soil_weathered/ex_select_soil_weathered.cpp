#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  hmap::Vec2<int>   shape = {256, 256};
  hmap::Vec2<float> kw = {4.f, 4.f};
  uint              seed = 0;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  hmap::clamp_min_smooth(z, -0.2f, 0.1f);
  hmap::remap(z);

  int         ir_curvature = 0;
  int         ir_gradient = 4;
  hmap::Array c = hmap::gpu::select_soil_weathered(
      z,
      ir_curvature,
      ir_gradient,
      hmap::ClampMode::POSITIVE_ONLY,
      1.f,
      1.f,
      0.2f);

  hmap::remap(z);
  hmap::remap(c);

  hmap::export_banner_png("ex_select_soil_weathered.png",
                          {z, c},
                          hmap::Cmap::JET);
}
