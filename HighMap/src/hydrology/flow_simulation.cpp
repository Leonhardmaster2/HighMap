/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "highmap/internal/opencl_run.hpp"

#include "highmap/array.hpp"
#include "highmap/gpu/metal.hpp"
#include "highmap/hydrology/hydrology.hpp"
#include "highmap/internal/validation.hpp"
#include "highmap/range.hpp"

namespace hmap::gpu
{

float helper_vflow_compute_adaptive_dt(const Array &h,
                                       float        viscosity,
                                       float        power)
{
  float hmax = h.max();
  float href = std::max(hmax, 0.1f);
  float mobility = std::pow(href, power) / viscosity;
  float c = 0.2f; // safety factor
  return c / fmax(mobility, 1e-6f);
}

Array flow_simulation(const Array &z,
                      float        water_height,
                      const Array &depth_map,
                      int          iterations,
                      float        dt,
                      bool         flux_diffusion,
                      float        flux_diffusion_strength,
                      float        dry_out_ratio,
                      const Array *p_rain_map,
                      float        rain_rate,
                      float        evap_rate,
                      bool         outflow_boundaries,
                      Array       *p_vel_u,
                      Array       *p_vel_v)
{
  if (!validate_non_empty(z)) return Array();
  if (!validate_same_shape(z, depth_map)) return Array();
  if (p_rain_map && !validate_same_shape(z, *p_rain_map)) return Array();

  const glm::ivec2 shape = z.shape;

  Array d = water_height * depth_map;
  Array u(shape);
  Array v(shape);

  auto finalize = [&]()
  {
    if (p_vel_u) *p_vel_u = u;
    if (p_vel_v) *p_vel_v = v;

    // remove thin layer of remaining water
    if (dry_out_ratio != 0.f)
    {
      float dmax = d.max();
      water_depth_dry_out(d, dry_out_ratio, nullptr, dmax);
    }
    return d; // water depth
  };

  if (iterations <= 0) return finalize();

  if (metal::is_available())
  {
    metal::flow_simulation(z,
                           d,
                           iterations,
                           dt,
                           water_height,
                           flux_diffusion,
                           flux_diffusion_strength,
                           p_rain_map,
                           rain_rate,
                           evap_rate,
                           outflow_boundaries,
                           &u,
                           &v);
    return finalize();
  }

  // --- device state: depth and the four fluxes are ping-pong pairs (A/B)
  // that stay on the GPU for the whole loop; only the terrain is uploaded
  // once and only the results are read back at the end.

  Array zeros(shape);

  const bool use_rain = rain_rate > 0.f;
  const bool use_map = use_rain && p_rain_map;

  using clwrapper::Direction;

  // flux pass: (z, fl, fr, ft, fb, d1, fl_out, fr_out, ft_out, fb_out, ...)
  auto run_fp = clwrapper::Run("hydraulic_vpipes_flow_pass");

  run_fp.bind_imagef("z", z.vector, shape.x, shape.y);
  run_fp.bind_imagef("fl_a", zeros.vector, shape.x, shape.y, Direction::INOUT);
  run_fp.bind_imagef("fr_a", zeros.vector, shape.x, shape.y, Direction::INOUT);
  run_fp.bind_imagef("ft_a", zeros.vector, shape.x, shape.y, Direction::INOUT);
  run_fp.bind_imagef("fb_a", zeros.vector, shape.x, shape.y, Direction::INOUT);
  run_fp.bind_imagef("d_a", d.vector, shape.x, shape.y, Direction::INOUT);
  run_fp.bind_imagef("fl_b", zeros.vector, shape.x, shape.y, Direction::INOUT);
  run_fp.bind_imagef("fr_b", zeros.vector, shape.x, shape.y, Direction::INOUT);
  run_fp.bind_imagef("ft_b", zeros.vector, shape.x, shape.y, Direction::INOUT);
  run_fp.bind_imagef("fb_b", zeros.vector, shape.x, shape.y, Direction::INOUT);

  run_fp.bind_arguments(shape.x,
                        shape.y,
                        dt,
                        flux_diffusion ? 1 : 0,
                        flux_diffusion_strength,
                        outflow_boundaries ? 1 : 0);

  // water pass: (z, fl, fr, ft, fb, d1, d2_out, u_out, v_out, ...)
  auto run_wa = clwrapper::Run("hydraulic_vpipes_water_pass",
                               run_fp.get_queue());

  run_wa.bind_image2d("z", run_fp.get_image2d("z"));
  run_wa.bind_image2d("fl_b", run_fp.get_image2d("fl_b"));
  run_wa.bind_image2d("fr_b", run_fp.get_image2d("fr_b"));
  run_wa.bind_image2d("ft_b", run_fp.get_image2d("ft_b"));
  run_wa.bind_image2d("fb_b", run_fp.get_image2d("fb_b"));
  run_wa.bind_image2d("d_a", run_fp.get_image2d("d_a"));
  run_wa.bind_imagef("d_b", d.vector, shape.x, shape.y, Direction::INOUT);
  run_wa.bind_imagef("u", u.vector, shape.x, shape.y, Direction::OUT);
  run_wa.bind_imagef("v", v.vector, shape.x, shape.y, Direction::OUT);

  run_wa.bind_arguments(shape.x,
                        shape.y,
                        dt,
                        water_height,
                        evap_rate,
                        outflow_boundaries ? 1 : 0);

  // rain pass: (d_in, rain_map, d_out, nx, ny, amount, use_map)
  std::unique_ptr<clwrapper::Run> run_rain;

  if (use_rain)
  {
    run_rain = std::make_unique<clwrapper::Run>("hydraulic_vpipes_rain_pass",
                                                run_fp.get_queue());

    run_rain->bind_image2d("d_a", run_fp.get_image2d("d_a"));
    run_rain->bind_imagef("rain",
                          use_map ? p_rain_map->vector : zeros.vector,
                          shape.x,
                          shape.y);
    run_rain->bind_image2d("d_b", run_wa.get_image2d("d_b"));
    run_rain->bind_arguments(shape.x, shape.y, rain_rate * dt, use_map ? 1 : 0);
  }

  // ping-pong handles
  const std::array<cl::Image2D, 2> img_d = {run_fp.get_image2d("d_a").cl_image,
                                            run_wa.get_image2d("d_b").cl_image};

  const std::array<std::array<cl::Image2D, 4>, 2> img_f = {
      {{run_fp.get_image2d("fl_a").cl_image,
        run_fp.get_image2d("fr_a").cl_image,
        run_fp.get_image2d("ft_a").cl_image,
        run_fp.get_image2d("fb_a").cl_image},
       {run_fp.get_image2d("fl_b").cl_image,
        run_fp.get_image2d("fr_b").cl_image,
        run_fp.get_image2d("ft_b").cl_image,
        run_fp.get_image2d("fb_b").cl_image}}};

  int dc = 0; // index of the current depth image
  int fc = 0; // index of the current flux images

  for (int it = 0; it < iterations; ++it)
  {
    // continuous rainfall: d[dc] + rain -> d[1 - dc]
    if (use_rain)
    {
      run_rain->set_argument(0, img_d[dc]);
      run_rain->set_argument(2, img_d[1 - dc]);
      run_rain->execute_async({shape.x, shape.y});
      dc = 1 - dc;
    }

    // flux update: reads f[fc], d[dc]; writes f[1 - fc]
    for (int k = 0; k < 4; ++k)
      run_fp.set_argument(1 + k, img_f[fc][k]);
    run_fp.set_argument(5, img_d[dc]);
    for (int k = 0; k < 4; ++k)
      run_fp.set_argument(6 + k, img_f[1 - fc][k]);

    run_fp.execute_async({shape.x, shape.y});

    // water transport: reads f[1 - fc], d[dc]; writes d[1 - dc], u, v
    for (int k = 0; k < 4; ++k)
      run_wa.set_argument(1 + k, img_f[1 - fc][k]);
    run_wa.set_argument(5, img_d[dc]);
    run_wa.set_argument(6, img_d[1 - dc]);

    run_wa.execute_async({shape.x, shape.y});

    fc = 1 - fc;
    dc = 1 - dc;
  }

  run_wa.finish();

  // retrieve results (both depth images map to the host array `d`)
  run_wa.read_imagef(dc == 0 ? "d_a" : "d_b");

  if (p_vel_u) run_wa.read_imagef("u");
  if (p_vel_v) run_wa.read_imagef("v");

  return finalize();
}

namespace
{

// box average for integer factors (no aliasing of fine terrain features),
// bilinear resampling otherwise
Array downsample(const Array &array, int factor, glm::ivec2 new_shape)
{
  if (array.shape.x != new_shape.x * factor || array.shape.y != new_shape.y * factor)
    return array.resample_to_shape(new_shape);

  Array       out(new_shape);
  const float norm = 1.f / float(factor * factor);

  for (int j = 0; j < new_shape.y; ++j)
    for (int i = 0; i < new_shape.x; ++i)
    {
      float sum = 0.f;
      for (int q = 0; q < factor; ++q)
        for (int p = 0; p < factor; ++p)
          sum += array(i * factor + p, j * factor + q);
      out(i, j) = sum * norm;
    }
  return out;
}

} // namespace

Array flow_simulation_coarse_to_fine(const Array &z,
                                     float        water_height,
                                     const Array &depth_map,
                                     int          iterations,
                                     int          coarse_factor,
                                     float        refine_ratio,
                                     float        dt,
                                     bool         flux_diffusion,
                                     float        flux_diffusion_strength,
                                     float        dry_out_ratio,
                                     const Array *p_rain_map,
                                     float        rain_rate,
                                     float        evap_rate,
                                     bool         outflow_boundaries)
{
  if (!validate_non_empty(z)) return Array();
  if (!validate_same_shape(z, depth_map)) return Array();
  if (p_rain_map && !validate_same_shape(z, *p_rain_map)) return Array();

  const glm::ivec2 shape = z.shape;
  const glm::ivec2 shape_c = shape / std::max(coarse_factor, 1);

  // nothing to gain below a useful coarse grid: reference solver
  if (coarse_factor <= 1 || std::min(shape_c.x, shape_c.y) < 32)
    return flow_simulation(z,
                           water_height,
                           depth_map,
                           iterations,
                           dt,
                           flux_diffusion,
                           flux_diffusion_strength,
                           dry_out_ratio,
                           p_rain_map,
                           rain_rate,
                           evap_rate,
                           outflow_boundaries);

  // (1) full simulated duration on the coarse grid: the iteration count
  // scales with the resolution (same physical time per cell), so this costs
  // ~1 / coarse_factor^3 of the reference run
  const Array z_c = downsample(z, coarse_factor, shape_c);
  const Array depth_c = downsample(depth_map, coarse_factor, shape_c);
  const Array rain_c = p_rain_map ? downsample(*p_rain_map, coarse_factor, shape_c)
                                  : Array();
  const int   iterations_c = std::max(
      1,
      static_cast<int>(std::lround(float(iterations) * shape_c.x / shape.x)));

  const Array d_c = flow_simulation(z_c,
                                    water_height,
                                    depth_c,
                                    iterations_c,
                                    dt,
                                    flux_diffusion,
                                    flux_diffusion_strength,
                                    /* dry_out_ratio */ 0.f,
                                    p_rain_map ? &rain_c : nullptr,
                                    rain_rate,
                                    evap_rate,
                                    outflow_boundaries);

  // (2) water depth back on the fine grid: dry coarse cells stay dry, so
  // fine-scale pits outside the flow do not fill up
  Array d_fine = d_c.resample_to_shape_bilinear(shape);
  clamp_min(d_fine, 0.f);

  // (3) short fine-resolution refinement, starting from that water: it only
  // has to settle the water over a few coarse cells onto the fine terrain
  const int iterations_f = std::max(
      1,
      static_cast<int>(std::lround(std::clamp(refine_ratio, 0.f, 1.f) * iterations)));

  return flow_simulation(z,
                         /* water_height: d_fine is the initial depth */ 1.f,
                         d_fine,
                         iterations_f,
                         dt,
                         flux_diffusion,
                         flux_diffusion_strength,
                         dry_out_ratio,
                         p_rain_map,
                         rain_rate,
                         evap_rate,
                         outflow_boundaries);
}

Array flow_simulation_viscous(const Array &z,
                              float        water_height,
                              const Array &depth_map,
                              int          iterations,
                              float        dt,
                              float        dry_out_ratio,
                              float        viscosity,
                              float        power,
                              float        evap_rate,
                              bool         outflow_boundaries)
{
  if (!validate_non_empty(z)) return Array();
  if (!validate_same_shape(z, depth_map)) return Array();

  const glm::ivec2 shape = z.shape;

  Array d = water_height * depth_map;

  auto finalize = [&]()
  {
    // remove thin layer of remaining water
    if (dry_out_ratio != 0.f)
    {
      float dmax = d.max();
      water_depth_dry_out(d, dry_out_ratio, nullptr, dmax);
    }
    return d; // depth
  };

  if (iterations <= 0) return finalize();

  // --- Device state: depth is a ping-pong pair (A/B) on GPU

  using clwrapper::Direction;

  if (dt <= 0.f)
    dt = helper_vflow_compute_adaptive_dt(d, viscosity, power);
  else
    dt = std::min(dt, helper_vflow_compute_adaptive_dt(d, viscosity, power));

  auto run = clwrapper::Run("shallow_viscous_flow");

  run.bind_imagef("z", z.vector, shape.x, shape.y);
  run.bind_imagef("d_a", d.vector, shape.x, shape.y, Direction::INOUT);
  run.bind_imagef("d_b", d.vector, shape.x, shape.y, Direction::INOUT);

  run.bind_arguments(shape.x,
                     shape.y,
                     dt,
                     viscosity,
                     power,
                     evap_rate,
                     outflow_boundaries ? 1 : 0);

  // ping-pong handles
  const std::array<cl::Image2D, 2> img_d = {run.get_image2d("d_a").cl_image,
                                            run.get_image2d("d_b").cl_image};

  int dc = 0; // index of the current depth image

  for (int it = 0; it < iterations; ++it)
  {
    run.set_argument(1, img_d[dc]);
    run.set_argument(2, img_d[1 - dc]);

    run.execute_async({shape.x, shape.y});

    dc = 1 - dc;
  }

  run.finish();

  // retrieve results (both depth images map to the host array `d`)
  run.read_imagef(dc == 0 ? "d_a" : "d_b");

  return finalize();
}

} // namespace hmap::gpu
