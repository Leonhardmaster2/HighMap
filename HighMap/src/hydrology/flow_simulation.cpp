/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "highmap/filters.hpp"
#include "highmap/math.hpp"
#include "highmap/opencl/gpu_opencl.hpp"
#include "highmap/range.hpp"

namespace hmap::gpu
{

Array flow_simulation(const Array &z,
                      float        water_height,
                      const Array &depth_map,
                      int          iterations,
                      float        dt,
                      bool         flux_diffusion,
                      float        flux_diffusion_strength,
                      float        dry_out_ratio)
{
  glm::ivec2 shape = z.shape;

  Array d = water_height * depth_map;

  Array fl(shape); // left flux
  Array fr(shape); // right
  Array ft(shape); // top
  Array fb(shape); // bottom

  for (int it = 0; it < iterations; ++it)
  {

    // --- flux update

    auto run_fp = clwrapper::Run("hydraulic_vpipes_flow_pass");

    run_fp.bind_imagef("z", z.vector, shape.x, shape.y); // inputs
    run_fp.bind_imagef("fl", fl.vector, shape.x, shape.y);
    run_fp.bind_imagef("fr", fr.vector, shape.x, shape.y);
    run_fp.bind_imagef("ft", ft.vector, shape.x, shape.y);
    run_fp.bind_imagef("fb", fb.vector, shape.x, shape.y);
    run_fp.bind_imagef("d1", d.vector, shape.x, shape.y);

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
    run_wa.bind_imagef("d1", d.vector, shape.x, shape.y);

    Array u(shape), v(shape);

    run_wa.bind_imagef("d2_out", d.vector, shape.x, shape.y, true); // outputs
    run_wa.bind_imagef("u_out", u.vector, shape.x, shape.y, true);
    run_wa.bind_imagef("v_out", v.vector, shape.x, shape.y, true);

    run_wa.bind_arguments(shape.x, shape.y, dt, water_height);

    run_wa.execute({shape.x, shape.y});

    run_wa.read_imagef("d2_out");
    run_wa.read_imagef("u_out");
    run_wa.read_imagef("v_out");
  }

  // remove thin layer of remaining water
  if (dry_out_ratio != 0.f)
  {
    float dmax = d.max();
    water_depth_dry_out(d, dry_out_ratio, nullptr, dmax);
  }

  return d; // water depth
}

} // namespace hmap::gpu
