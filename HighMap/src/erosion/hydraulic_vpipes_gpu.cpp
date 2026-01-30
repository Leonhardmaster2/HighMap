/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "macrologger.h"

#include "highmap/boundary.hpp"
#include "highmap/filters.hpp"
#include "highmap/math.hpp"
#include "highmap/opencl/gpu_opencl.hpp"
#include "highmap/range.hpp"

#include "highmap/selector.hpp"

namespace hmap::gpu
{

void hydraulic_vpipes(Array &z,
                      float  water_height,
                      bool   maintain_water_volume,
                      float  evap_rate,
                      int    iterations,
                      float  dt,
                      float  k_capacity,
                      float  k_erode,
                      float  k_depose,
                      float  k_discharge_exp,
                      float  downcutting_max_depth_ratio,
                      bool   flux_diffusion,
                      float  flux_diffusion_strength,
                      Array *p_rain_map,
                      Array *p_water_depth,
                      Array *p_sediment,
                      Array *p_vel_u,
                      Array *p_vel_v)
{
  // TODO DBG REMOVE
  water_height = 1e-2f;
  k_capacity = 0.5f;
  downcutting_max_depth_ratio = 2.5f;

  bool add_rain_map_noise = true;
  float spl_erosion_strength = 10.f;
  
  //

  const glm::ivec2 shape = z.shape;

  Array rain_map(shape, 1.f);
  if (p_rain_map) rain_map = *p_rain_map;

  // TODO DBG REMOVE

  // rain_map = cubic_pulse(shape, nullptr, nullptr);

  if (add_rain_map_noise)
    rain_map *= white(shape, 0.f, 1.f, 0);

  Array d(shape, water_height); // water height
  Array d1(shape);
  Array d2(shape);

  Array s(shape); // sediment height

  Array fl(shape); // left flux
  Array fr(shape); // right
  Array ft(shape); // top
  Array fb(shape); // bottom

  Array u(shape);
  Array v(shape);

  d *= rain_map;
  const float water_volume_init = d.sum();
  const float rain_map_volume = rain_map.sum();

  // --- main loop

  for (int it = 0; it < iterations; ++it)
  {

    // ---

    // TODO tune, parametrize
    gpu::hydraulic_schott_erosion(z,
                                  1,
                                  spl_erosion_strength,
                                  0.5f,
                                  1.3f,
                                  &rain_map);
    // gpu::thermal(z, &rain_map, Array(shape, 1.f / shape.x), 1);

    // --- water increase (CPU)

    d1 = d; // TODO kept as reference, TO REMOVE

    // --- flux update

    auto run_fp = clwrapper::Run("hydraulic_vpipes_flow_pass");

    run_fp.bind_imagef("z", z.vector, shape.x, shape.y); // inputs
    run_fp.bind_imagef("fl", fl.vector, shape.x, shape.y);
    run_fp.bind_imagef("fr", fr.vector, shape.x, shape.y);
    run_fp.bind_imagef("ft", ft.vector, shape.x, shape.y);
    run_fp.bind_imagef("fb", fb.vector, shape.x, shape.y);
    run_fp.bind_imagef("d1", d1.vector, shape.x, shape.y);

    run_fp.bind_imagef("fl_out", fl.vector, shape.x, shape.y, true); // outputs
    run_fp.bind_imagef("fr_out", fr.vector, shape.x, shape.y, true);
    run_fp.bind_imagef("ft_out", ft.vector, shape.x, shape.y, true);
    run_fp.bind_imagef("fb_out", fb.vector, shape.x, shape.y, true);

    run_fp.bind_arguments(shape.x,
                          shape.y,
                          dt,
                          flux_diffusion ? 1 : 0,
                          flux_diffusion_strength);

    run_fp.execute({shape.x, shape.y});

    // update flux (from GPU to CPU)
    run_fp.read_imagef("fl_out");
    run_fp.read_imagef("fr_out");
    run_fp.read_imagef("ft_out");
    run_fp.read_imagef("fb_out");

    // --- water transport

    auto run_wa = clwrapper::Run("hydraulic_vpipes_water_pass");

    run_wa.bind_imagef("z", z.vector, shape.x, shape.y); // inputs
    run_wa.bind_imagef("fl", fl.vector, shape.x, shape.y);
    run_wa.bind_imagef("fr", fr.vector, shape.x, shape.y);
    run_wa.bind_imagef("ft", ft.vector, shape.x, shape.y);
    run_wa.bind_imagef("fb", fb.vector, shape.x, shape.y);
    run_wa.bind_imagef("d1", d1.vector, shape.x, shape.y);

    run_wa.bind_imagef("d2_out", d2.vector, shape.x, shape.y, true); // outputs
    run_wa.bind_imagef("u_out", u.vector, shape.x, shape.y, true);
    run_wa.bind_imagef("v_out", v.vector, shape.x, shape.y, true);

    run_wa.bind_arguments(shape.x, shape.y, dt, water_height);

    run_wa.execute({shape.x, shape.y});

    run_wa.read_imagef("d2_out");
    run_wa.read_imagef("u_out");
    run_wa.read_imagef("v_out");

    // --- erosion and deposition

    auto run_er = clwrapper::Run("hydraulic_vpipes_erosion_pass");

    run_er.bind_imagef("z", z.vector, shape.x, shape.y); // inputs
    run_er.bind_imagef("d2", d2.vector, shape.x, shape.y);
    run_er.bind_imagef("u", u.vector, shape.x, shape.y);
    run_er.bind_imagef("v", v.vector, shape.x, shape.y);
    run_er.bind_imagef("s", s.vector, shape.x, shape.y);

    run_er.bind_imagef("z_out", z.vector, shape.x, shape.y, true); // outputs
    run_er.bind_imagef("s_out", s.vector, shape.x, shape.y, true);

    run_er.bind_arguments(shape.x,
                          shape.y,
                          water_height,
                          k_capacity,
                          k_erode,
                          k_depose,
                          k_discharge_exp,
                          downcutting_max_depth_ratio);

    run_er.execute({shape.x, shape.y});

    run_er.read_imagef("z_out");
    run_er.read_imagef("s_out");

    // --- sediment transport

    auto run_st = clwrapper::Run("hydraulic_vpipes_sediment_transport_pass");

    run_st.bind_imagef("u", u.vector, shape.x, shape.y); // inputs
    run_st.bind_imagef("v", v.vector, shape.x, shape.y);
    run_st.bind_imagef("s", s.vector, shape.x, shape.y);

    run_st.bind_imagef("s_out", s.vector, shape.x, shape.y, true); // outputs

    run_st.bind_arguments(shape.x, shape.y, dt);

    run_st.execute({shape.x, shape.y});

    run_st.read_imagef("s_out");

    // --- water evaporation

    d = d2 * (1.f - dt * evap_rate);

    if (maintain_water_volume)
    {
      float water_to_add = water_volume_init - d.sum();
      d += water_to_add * rain_map / rain_map_volume;
    }
  }

  // --- optional outputs

  if (p_water_depth) *p_water_depth = d;
  if (p_sediment) *p_sediment = s;
  if (p_vel_u) *p_vel_u = u;
  if (p_vel_v) *p_vel_v = v;
}

} // namespace hmap::gpu
