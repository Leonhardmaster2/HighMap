/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/boundary.hpp"
#include "highmap/filters.hpp"
#include "highmap/math.hpp"
#include "highmap/opencl/gpu_opencl.hpp"
#include "highmap/operator.hpp"
#include "highmap/range.hpp"

namespace hmap::gpu
{

void hydraulic_particle(Array       &z,
                        int          nparticles,
                        uint         seed,
                        const Array *p_bedrock,
                        const Array *p_moisture_map,
                        const Array *p_elevation_shift,
                        Array       *p_erosion_map,
                        Array       *p_deposition_map,
                        float        c_capacity,
                        float        c_erosion,
                        float        c_deposition,
                        float        c_inertia,
                        float        c_gravity,
                        float        drag_rate,
                        float        evap_rate,
                        bool         enable_directional_bias,
                        float        angle_bias)
{
  const glm::ivec2 shape = z.shape;

  Array z_bckp = Array();
  if ((p_erosion_map != nullptr) || (p_deposition_map != nullptr)) z_bckp = z;

  // kernel
  auto run = clwrapper::Run("hydraulic_particle");

  run.bind_buffer<float>("z", z.vector);
  helper_bind_optional_buffer(run, "bedrock", p_bedrock);
  helper_bind_optional_buffer(run, "moisture_map", p_moisture_map);
  helper_bind_optional_buffer(run, "elevation_shift", p_elevation_shift);

  run.bind_arguments(shape.x,
                     shape.y,
                     nparticles,
                     seed,
                     c_capacity,
                     c_erosion,
                     c_deposition,
                     c_inertia,
                     c_gravity,
                     drag_rate,
                     evap_rate,
                     enable_directional_bias ? 1 : 0,
                     angle_bias,
                     p_bedrock ? 1 : 0,
                     p_moisture_map ? 1 : 0,
                     p_elevation_shift ? 1 : 0);

  run.write_buffer("z");
  run.execute(nparticles);
  run.read_buffer("z");

  // --- post-treatments

  extrapolate_borders(z);

  // splatmaps
  if (p_erosion_map)
  {
    *p_erosion_map = z_bckp - z;
    clamp_min(*p_erosion_map, 0.f);
  }

  if (p_deposition_map)
  {
    *p_deposition_map = z - z_bckp;
    clamp_min(*p_deposition_map, 0.f);
  }
}

void hydraulic_particle(Array       &z,
                        const Array *p_mask,
                        int          nparticles,
                        uint         seed,
                        const Array *p_bedrock,
                        const Array *p_moisture_map,
                        const Array *p_elevation_shift,
                        Array       *p_erosion_map,
                        Array       *p_deposition_map,
                        float        c_capacity,
                        float        c_erosion,
                        float        c_deposition,
                        float        c_inertia,
                        float        c_gravity,
                        float        drag_rate,
                        float        evap_rate,
                        bool         enable_directional_bias,
                        float        angle_bias)
{
  apply_with_mask(z,
                  p_mask,
                  [&](Array &a)
                  {
                    gpu::hydraulic_particle(a,
                                            nparticles,
                                            seed,
                                            p_bedrock,
                                            p_moisture_map,
                                            p_elevation_shift,
                                            p_erosion_map,
                                            p_deposition_map,
                                            c_capacity,
                                            c_erosion,
                                            c_deposition,
                                            c_inertia,
                                            c_gravity,
                                            drag_rate,
                                            evap_rate,
                                            enable_directional_bias,
                                            angle_bias);
                  });
}

} // namespace hmap::gpu
