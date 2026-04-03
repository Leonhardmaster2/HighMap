#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  glm::vec2  kw = {4.f, 4.f};
  int        seed = 2;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, kw, seed);
  z = hmap::bulkify(z, hmap::PrimitiveType::PRIM_CUBIC_PULSE, 1.f);
  hmap::remap(z);

  // parameters
  const float       talus_repose = 2.5f / shape.x;
  const int         iterations = 500;
  const float       amount = 1e-1f;
  const hmap::Array talus(shape, talus_repose);

  // no-melting
  hmap::Array snow_depth0 = hmap::gpu::snow_simulation(
      z,
      amount,
      hmap::Array(shape, 1.f), // fall_map
      hmap::Array(shape, 0.f), // melting_map
      talus,
      iterations);

  // with melting
  hmap::Array melt_map = hmap::snow_melting_map(z);

  hmap::Array snow_depth1 = hmap::gpu::snow_simulation(
      z,
      amount,
      hmap::Array(shape, 1.f), // fall_map
      melt_map,
      talus,
      iterations);

  z.dump("z0.png");
  (z + snow_depth1).dump("z.png");
  snow_depth1.dump("depth.png");

  hmap::export_banner_png("ex_snow_simulation.png",
                          {z, z + snow_depth0, melt_map, z + snow_depth1},
                          hmap::Cmap::JET);
}
