/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

#include <vector>

#include "cl_wrapper/run.hpp"

#include "highmap/array.hpp"
#include "highmap/erosion.hpp"
#include "highmap/internal/validation.hpp"

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
  if (!validate_non_empty(z)) return;
  if (p_rain_map && !validate_same_shape(z, *p_rain_map)) return;

  const glm::ivec2 shape = z.shape;

  Array rain_map(shape, 1.f);
  if (p_rain_map) rain_map = *p_rain_map;

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
    // (1) water volume increment
    if (maintain_water_volume)
    {
      float water_volume = d.sum();
      float rain_rate = (water_volume_init - water_volume) / rain_map_volume;
      d += rain_rate * rain_map;
    }

    // (2) flow simulation
    {
      auto run = clwrapper::Run("hydraulic_vpipes_flow_simulation");

      run.bind_buffer<float>("z", z.vector);
      run.bind_buffer<float>("d", d.vector);
      run.bind_buffer<float>("fl", fl.vector);
      run.bind_buffer<float>("fr", fr.vector);
      run.bind_buffer<float>("ft", ft.vector);
      run.bind_buffer<float>("fb", fb.vector);
      run.bind_buffer<float>("d1", d1.vector);

      run.bind_arguments(shape.x, shape.y, dt);

      run.write_buffer("z");
      run.write_buffer("d");
      run.write_buffer("fl");
      run.write_buffer("fr");
      run.write_buffer("ft");
      run.write_buffer("fb");

      run.execute({shape.x, shape.y});

      run.read_buffer("fl");
      run.read_buffer("fr");
      run.read_buffer("ft");
      run.read_buffer("fb");
      run.read_buffer("d1");
    }

    // (3) calculate velocity field, update water level and apply
    // evaporation
    {
      auto run = clwrapper::Run("hydraulic_vpipes_velocity");

      run.bind_buffer<float>("d", d.vector);
      run.bind_buffer<float>("fl", fl.vector);
      run.bind_buffer<float>("fr", fr.vector);
      run.bind_buffer<float>("ft", ft.vector);
      run.bind_buffer<float>("fb", fb.vector);
      run.bind_buffer<float>("d1", d1.vector);
      run.bind_buffer<float>("u", u.vector);
      run.bind_buffer<float>("v", v.vector);
      run.bind_buffer<float>("d2", d2.vector);

      run.bind_arguments(shape.x,
                         shape.y,
                         dt,
                         evap_rate,
                         flux_diffusion ? 1 : 0,
                         flux_diffusion_strength);

      run.write_buffer("d");
      run.write_buffer("fl");
      run.write_buffer("fr");
      run.write_buffer("ft");
      run.write_buffer("fb");
      run.write_buffer("d1");

      run.execute({shape.x, shape.y});

      run.read_buffer("u");
      run.read_buffer("v");
      run.read_buffer("d2");
      run.read_buffer("fl");
      run.read_buffer("fr");
      run.read_buffer("ft");
      run.read_buffer("fb");
    }

    // (4) erosion & deposition
    {
      auto run = clwrapper::Run("hydraulic_vpipes_erosion_deposition");

      run.bind_buffer<float>("z", z.vector);
      run.bind_buffer<float>("s", s.vector);
      run.bind_buffer<float>("d2", d2.vector);
      run.bind_buffer<float>("fl", fl.vector);
      run.bind_buffer<float>("fr", fr.vector);
      run.bind_buffer<float>("ft", ft.vector);
      run.bind_buffer<float>("fb", fb.vector);
      run.bind_buffer<float>("u", u.vector);
      run.bind_buffer<float>("v", v.vector);

      run.bind_arguments(shape.x,
                         shape.y,
                         dt,
                         k_capacity,
                         k_erode,
                         k_depose,
                         k_discharge_exp,
                         downcutting_max_depth_ratio);

      run.write_buffer("z");
      run.write_buffer("s");
      run.write_buffer("d2");
      run.write_buffer("fl");
      run.write_buffer("fr");
      run.write_buffer("ft");
      run.write_buffer("fb");
      run.write_buffer("u");
      run.write_buffer("v");

      run.execute({shape.x, shape.y});

      run.read_buffer("z");
      run.read_buffer("s");
    }

    // (5) sediment transportation (advection)
    {
      auto run = clwrapper::Run("hydraulic_vpipes_advection");

      run.bind_buffer<float>("s", s.vector);
      run.bind_buffer<float>("u", u.vector);
      run.bind_buffer<float>("v", v.vector);

      run.bind_arguments(shape.x, shape.y, dt);

      run.write_buffer("s");
      run.write_buffer("u");
      run.write_buffer("v");

      run.execute({shape.x, shape.y});

      run.read_buffer("s");
    }

    // state variable update
    d = d2;
  }

  // --- outputs

  if (p_water_depth) *p_water_depth = d;
  if (p_sediment) *p_sediment = s;
  if (p_vel_u) *p_vel_u = u;
  if (p_vel_v) *p_vel_v = v;
}

} // namespace hmap::gpu
