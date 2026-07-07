#include <iostream>

#include "highmap.hpp"

int main(void)
{
  glm::ivec2 shape = {256, 256};
  glm::vec2  res = {2.f, 2.f};
  int        seed = 1;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z);

  hmap::Array z0 = z;
  hmap::Array sediment, discharge;

  hmap::gpu::init_opencl();
  hmap::gpu::hydraulic_mcdonald(z, 256, 1, &sediment, &discharge);

  std::cout << "z min/max: " << z.min() << " " << z.max() << "\n";
  std::cout << "discharge min/max: " << discharge.min() << " "
            << discharge.max() << "\n";

  z0.to_png("ex_hydraulic_mcdonald0.png", hmap::Cmap::TERRAIN, true);
  z.to_png("ex_hydraulic_mcdonald1.png", hmap::Cmap::TERRAIN, true);

  // log-scale the discharge for visibility (spans orders of magnitude)
  hmap::Array dlog = discharge;
  for (auto &v : dlog.vector)
    v = std::log10(1.f + v);
  dlog.to_png("ex_hydraulic_mcdonald2.png", hmap::Cmap::HOT);

  hmap::Array zm = hmap::noise_fbm(hmap::NoiseType::PERLIN, {512, 512}, res, seed);
  hmap::remap(zm);
  hmap::Array discharge_m;
  hmap::gpu::hydraulic_mcdonald_multiscale(zm, 1, {256, 128, 64}, nullptr, &discharge_m);
  std::cout << "multiscale z min/max: " << zm.min() << " " << zm.max() << "\n";
  zm.to_png("ex_hydraulic_mcdonald3.png", hmap::Cmap::TERRAIN, true);
}
