#include <iostream>

#include "highmap.hpp"

int main(void)
{
  hmap::gpu::init_opencl();

  glm::ivec2 shape = {256, 256};
  // shape = {1024, 1024};
  glm::vec2 res = {3.f, 3.f};
  int       seed = 1;

  int steps = 64;

  hmap::Array z = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(z);

  hmap::Array z0 = z;
  hmap::Array sediment, discharge;

  hmap::gpu::McDonaldParams params = {.strength = 0.5f,
                                      .deposition = 0.5f,
                                      .crit_slope = 0.5f,
                                      .meandering = 1.f,
                                      .scale = 1.f,
                                      .relief_scale = 1.f};

  hmap::gpu::hydraulic_mcdonald(z,
                                steps,
                                seed,
                                params,
                                nullptr,
                                &sediment,
                                &discharge);

  // output
  {
    std::cout << "z min/max: " << z.min() << " " << z.max() << "\n";
    std::cout << "discharge min/max: " << discharge.min() << " "
              << discharge.max() << "\n";

    z0.to_png("ex_hydraulic_mcdonald0.png", hmap::Cmap::TERRAIN, true);
    z.to_png("ex_hydraulic_mcdonald1.png", hmap::Cmap::TERRAIN, true);
  }

  // log-scale the discharge for visibility (spans orders of magnitude)
  {
    hmap::Array dlog = discharge;
    for (auto &v : dlog.vector)
      v = std::log10(1.f + v);

    dlog.to_png("ex_hydraulic_mcdonald2.png", hmap::Cmap::HOT);
  }

  hmap::Array zm = hmap::noise_fbm(hmap::NoiseType::PERLIN, shape, res, seed);
  hmap::remap(zm);

  hmap::Array sediment_m, discharge_m;

  std::vector<int> steps_per_level = {steps / 4, steps / 2, steps};

  hmap::gpu::hydraulic_mcdonald_multiscale(zm,
                                           seed,
                                           steps_per_level,
                                           params,
                                           nullptr,
                                           &sediment_m,
                                           &discharge_m);

  std::cout << "multiscale z min/max: " << zm.min() << " " << zm.max() << "\n";
  zm.to_png("ex_hydraulic_mcdonald3.png", hmap::Cmap::TERRAIN, true);
}
