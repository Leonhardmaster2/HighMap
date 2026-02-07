/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/boundary.hpp"
#include "highmap/filters.hpp"
#include "highmap/gradient.hpp"
#include "highmap/math.hpp"
#include "highmap/opencl/gpu_opencl.hpp"
#include "highmap/range.hpp"

namespace hmap::gpu
{

Array snow_simulation(const Array &z,
                      float        snow_depth,
                      const Array &fall_map,
                      const Array &melting_map,
                      const Array &talus,
                      int          iterations,
                      float        dt,
                      float        fall_iterations_ratio,
                      float        k_snow,
                      float        k_visc,
                      float        k_melt_factor,
                      float        k_depth_ratio,
                      float        k_depth_slope_ratio,
                      bool         post_filter,
                      float        thermal_talus_ratio)
{
  const glm::ivec2 shape = z.shape;

  // output
  Array s(shape);

  // computed parameters
  const float k_melt = k_melt_factor / int(iterations);
  const int   iterations_fall = std::max(1,
                                       int(fall_iterations_ratio * iterations));

  // OpenCL kernel
  auto run = clwrapper::Run("snow_simulation");

  run.bind_imagef("z", z.vector, shape.x, shape.y); // inputs
  run.bind_imagef("s", s.vector, shape.x, shape.y);
  run.bind_imagef("talus", talus.vector, shape.x, shape.y);
  run.bind_imagef("melting_map", melting_map.vector, shape.x, shape.y);

  run.bind_imagef("s_out", s.vector, shape.x, shape.y, true); // outputs

  run.bind_arguments(shape.x,
                     shape.y,
                     dt,
                     snow_depth,
                     k_snow,
                     k_melt,
                     k_visc,
                     k_depth_ratio,
                     k_depth_slope_ratio);

  for (int it = 0; it < iterations; ++it)
  {
    if (it < iterations_fall) s += snow_depth / iterations_fall * fall_map;

    run.write_imagef("s");
    run.execute({shape.x, shape.y});
    run.read_imagef("s_out");

    fill_borders(s);
  }

  extrapolate_borders(s);

  if (post_filter)
  {
    Array mask = s / snow_depth;
    clamp_max_smooth(mask, 1.f);
    Array z_wrk = z + s;
    gpu::thermal(z_wrk,
                 &mask,
                 thermal_talus_ratio * hmap::gradient_norm(z_wrk),
                 iterations);
    s = z_wrk - z;
  }

  clamp_min(s, 0.f);

  return s;
}

} // namespace hmap::gpu
