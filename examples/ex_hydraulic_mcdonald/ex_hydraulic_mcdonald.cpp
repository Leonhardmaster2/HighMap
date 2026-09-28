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

  // parameters for hydraulic_mcdonald and hydraulic_mcdonald_multiscale
  std::vector<int> steps_per_level = {steps / 4, steps / 2, steps};
  float            world_extent_km = 40.f;
  float            z_scale_km = 4.f;
  int              samples = 8192;
  int              maxage = 512;
  float            lrate = 0.2f;
  float            time_step = 10.f;
  float            rainfall = 1.f;
  float            evap_rate = 1e-9f;
  float            gravity = 9.81f;
  float            viscosity = 0.025f;
  float            bed_shear = 0.01f;
  float            crit_slope = 0.5f;
  float            settle_rate = 0.1f;
  float            thermal_rate = 2.5e-3f;
  float            deposition_rate = 5e-3f;
  float            suspension_rate = 2.5e-4f;
  float            exit_slope = 0.05f;

  hmap::gpu::hydraulic_mcdonald(z,
                                steps,
                                seed,
                                &sediment,
                                &discharge,
                                world_extent_km,
                                z_scale_km,
                                samples,
                                maxage,
                                lrate,
                                time_step,
                                rainfall,
                                evap_rate,
                                gravity,
                                viscosity,
                                bed_shear,
                                crit_slope,
                                settle_rate,
                                thermal_rate,
                                deposition_rate,
                                suspension_rate,
                                exit_slope);

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

  hmap::gpu::hydraulic_mcdonald_multiscale(zm,
                                           seed,
                                           steps_per_level,
                                           &sediment_m,
                                           &discharge_m,
                                           world_extent_km,
                                           z_scale_km,
                                           samples,
                                           maxage,
                                           lrate,
                                           time_step,
                                           rainfall,
                                           evap_rate,
                                           gravity,
                                           viscosity,
                                           bed_shear,
                                           crit_slope,
                                           settle_rate,
                                           thermal_rate,
                                           deposition_rate,
                                           suspension_rate,
                                           exit_slope);

  std::cout << "multiscale z min/max: " << zm.min() << " " << zm.max() << "\n";
  zm.to_png("ex_hydraulic_mcdonald3.png", hmap::Cmap::TERRAIN, true);
}
